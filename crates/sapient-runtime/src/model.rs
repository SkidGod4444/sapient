//! `Model` — loaded model metadata and compiled IR.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use sapient_core::error::{Result, SapientError};
use sapient_ir::graph::Graph;
use sapient_ir::passes::PassManager;
use sapient_ir::shape_inference::ShapeRegistry;

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// Model loading and compilation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Run IR optimisation passes before execution (default: true).
    pub optimize: bool,
    /// Run shape inference before execution (default: true).
    pub infer_shapes: bool,
    /// Maximum batch size hint for pool pre-allocation.
    pub max_batch_size: usize,
    /// Backend name: "cpu", "metal", "vulkan" (default: "cpu").
    pub backend: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            optimize: true,
            infer_shapes: true,
            max_batch_size: 32,
            backend: "cpu".into(),
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// A loaded, optionally optimised model ready for session creation.
pub struct Model {
    pub name: String,
    pub graph: Graph,
    pub config: ModelConfig,
}

impl Model {
    /// Load a model from a file path (format auto-detected).
    pub fn load(path: &Path, config: ModelConfig) -> Result<Self> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_owned();

        info!(path = %path.display(), "loading model");

        let mut graph = sapient_io::load_graph(path)?;

        if config.infer_shapes {
            let mut reg = ShapeRegistry::new(&mut graph);
            reg.infer()?;
        }

        if config.optimize {
            let pm = PassManager::standard();
            pm.run_all(&mut graph)?;
        }

        info!(
            model = %name,
            nodes = graph.node_count(),
            "model loaded and compiled"
        );

        Ok(Self {
            name,
            graph,
            config,
        })
    }

    /// Load a model from raw ONNX bytes.
    pub fn from_onnx_bytes(bytes: &[u8], config: ModelConfig) -> Result<Self> {
        let mut graph = sapient_io::OnnxLoader::from_bytes(bytes)?;

        if config.infer_shapes {
            ShapeRegistry::new(&mut graph).infer()?;
        }
        if config.optimize {
            PassManager::standard().run_all(&mut graph)?;
        }

        Ok(Self {
            name: "onnx_model".into(),
            graph,
            config,
        })
    }

    /// Build a model from an already-constructed graph (e.g. in tests).
    pub fn from_graph(
        name: impl Into<String>,
        mut graph: Graph,
        config: ModelConfig,
    ) -> Result<Self> {
        if config.infer_shapes {
            ShapeRegistry::new(&mut graph).infer()?;
        }
        if config.optimize {
            PassManager::standard().run_all(&mut graph)?;
        }
        Ok(Self {
            name: name.into(),
            graph,
            config,
        })
    }
}
