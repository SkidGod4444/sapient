//! Network integration tests for Hub download performance settings.
//!
//! Run serially — parallel tests contend on HuggingFace Hub file locks for gpt2.

use sapient_hub::{HubClient, LoadOptions};

#[tokio::test]
async fn hub_download_fast_then_sequential() -> anyhow::Result<()> {
    let fast = HubClient::with_options(LoadOptions {
        formats: vec!["safetensors".into()],
        quiet: true,
        fast_download: true,
        ..Default::default()
    })?;

    let files = fast.download("gpt2").await?;
    assert!(files.config_path.exists());
    assert!(files
        .weight_paths
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "safetensors")));

    let sequential = HubClient::with_options(LoadOptions {
        formats: vec!["safetensors".into()],
        quiet: true,
        fast_download: false,
        ..Default::default()
    })?;

    let cached = sequential.download("gpt2").await?;
    assert!(!cached.weight_paths.is_empty());
    Ok(())
}
