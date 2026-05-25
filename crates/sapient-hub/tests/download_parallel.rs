//! Network integration tests for Hub download performance settings.

use sapient_hub::{HubClient, LoadOptions};

#[tokio::test]
async fn fast_download_fetches_public_model() -> anyhow::Result<()> {
    let client = HubClient::with_options(LoadOptions {
        formats: vec!["safetensors".into()],
        quiet: true,
        fast_download: true,
        ..Default::default()
    })?;

    let files = client.download("gpt2").await?;
    assert!(files.config_path.exists());
    assert!(
        files
            .weight_paths
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "safetensors"))
    );
    Ok(())
}

#[tokio::test]
async fn sequential_mode_uses_cache_after_fast_download() -> anyhow::Result<()> {
    // Run after fast_download_fetches_public_model (same process, cached gpt2).
    let client = HubClient::with_options(LoadOptions {
        formats: vec!["safetensors".into()],
        quiet: true,
        fast_download: false,
        ..Default::default()
    })?;

    let files = client.download("gpt2").await?;
    assert!(!files.weight_paths.is_empty());
    Ok(())
}
