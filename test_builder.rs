use hf_hub::api::tokio::ApiBuilder;
fn main() {
    let client = reqwest::ClientBuilder::new().http1_only().build().unwrap();
    let builder = ApiBuilder::new().with_client(client);
}
