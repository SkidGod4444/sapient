use std::collections::HashMap;

/// Resolves a short alias (like "phi-2") to a HuggingFace repository ID (like "microsoft/phi-2").
pub fn resolve_model_alias(alias: &str) -> String {
    let mut map = HashMap::new();
    map.insert("openhorizon/phi-2", "microsoft/phi-2");
    
    // Convert to lowercase to allow case-insensitive matching
    let normalized = alias.to_lowercase();
    
    if let Some(repo_id) = map.get(normalized.as_str()) {
        repo_id.to_string()
    } else {
        // If it's not an alias, just return the original input
        // which might already be a valid HuggingFace repo ID.
        alias.to_string()
    }
}
