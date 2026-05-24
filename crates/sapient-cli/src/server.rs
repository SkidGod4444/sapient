//! HTTP inference server using axum.
//!
//! Endpoints:
//!   POST /v1/infer          — single request
//!   POST /v1/batch_infer    — explicit batch
//!   GET  /v1/health         — health check
//!   GET  /v1/metrics        — Prometheus-compatible text metrics

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use sapient_core::Tensor;
use sapient_runtime::{InferenceSession, Model, ModelConfig, SessionOptions};

// ── AppState ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    session: Arc<InferenceSession>,
}

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct InferRequest {
    inputs: HashMap<String, TensorSpec>,
}

#[derive(Debug, Deserialize)]
struct BatchInferRequest {
    batch: Vec<HashMap<String, TensorSpec>>,
}

#[derive(Debug, Deserialize)]
struct TensorSpec {
    shape: Vec<usize>,
    data:  Vec<f32>,
}

#[derive(Debug, Serialize)]
struct InferResponse {
    outputs: Vec<TensorOut>,
    latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct TensorOut {
    shape: Vec<usize>,
    dtype: String,
    data:  Vec<f32>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    backend: String,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        backend: state.session.backend_name().to_owned(),
    })
}

async fn infer(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    let inputs: Result<HashMap<String, Tensor>, _> = req
        .inputs
        .into_iter()
        .map(|(k, v)| {
            Tensor::from_f32(&v.data, v.shape)
                .map(|t| (k, t))
                .map_err(|e| e.to_string())
        })
        .collect();

    let inputs = match inputs {
        Ok(i) => i,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e })),
            ).into_response();
        }
    };

    match state.session.run(inputs) {
        Ok(outputs) => {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            let resp = InferResponse {
                outputs: outputs.iter().map(tensor_to_out).collect(),
                latency_ms,
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn batch_infer(
    State(state): State<AppState>,
    Json(req): Json<BatchInferRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    let batch: Result<Vec<HashMap<String, Tensor>>, _> = req
        .batch
        .into_iter()
        .map(|item| {
            item.into_iter()
                .map(|(k, v)| {
                    Tensor::from_f32(&v.data, v.shape)
                        .map(|t| (k, t))
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<HashMap<_, _>, _>>()
        })
        .collect();

    let batch = match batch {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e })),
            ).into_response();
        }
    };

    let batch_size = batch.len();
    match state.session.run_batch(batch) {
        Ok(all_outputs) => {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            let results: Vec<_> = all_outputs
                .iter()
                .map(|outputs| {
                    serde_json::json!({
                        "outputs": outputs.iter().map(tensor_to_out).collect::<Vec<_>>()
                    })
                })
                .collect();
            (StatusCode::OK, Json(serde_json::json!({
                "batch_size": batch_size,
                "latency_ms": latency_ms,
                "results": results,
            }))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

async fn metrics_handler() -> impl IntoResponse {
    // Simple text response — integrate with metrics-exporter-prometheus for
    // full Prometheus scraping.
    (StatusCode::OK, "# SAPIENT metrics\n# (enable 'prometheus' feature for full export)\n")
}

// ── Server entry point ────────────────────────────────────────────────────────

pub async fn serve(
    model_path: PathBuf,
    port: u16,
    backend: String,
    _workers: usize,
) -> Result<()> {
    let config = ModelConfig { backend: backend.clone(), ..Default::default() };
    let model = Model::load(&model_path, config)?;
    let session = InferenceSession::new(
        model,
        SessionOptions { telemetry: true, ..Default::default() },
    )?;

    let state = AppState {
        session: Arc::new(session),
    };

    let app = Router::new()
        .route("/v1/health",       get(health))
        .route("/v1/infer",        post(infer))
        .route("/v1/batch_infer",  post(batch_infer))
        .route("/v1/metrics",      get(metrics_handler))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(addr = %addr, backend = %backend, "SAPIENT server starting");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn tensor_to_out(t: &Tensor) -> TensorOut {
    TensorOut {
        shape: t.shape().dims().to_vec(),
        dtype: t.dtype().to_string(),
        data:  if t.dtype() == sapient_core::DType::F32 {
            t.as_f32_slice().to_vec()
        } else {
            vec![]
        },
    }
}
