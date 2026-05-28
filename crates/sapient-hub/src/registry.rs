use std::collections::HashMap;
use anyhow::{bail, Result};

/// Resolves a short alias (like "openhorizon/phi-2") to a HuggingFace repository ID (like "microsoft/phi-2").
pub fn resolve_model_alias(alias: &str) -> Result<String> {
    let mut map = HashMap::new();
    map.insert("openhorizon/phi-2", "microsoft/phi-2");
    
    // Convert to lowercase to allow case-insensitive matching
    let normalized = alias.to_lowercase();
    
    if let Some(repo_id) = map.get(normalized.as_str()) {
        Ok(repo_id.to_string())
    } else {
        bail!("Model '{}' is not supported in the registry. Currently supported model is: openhorizon/phi-2", alias)
    }
}
