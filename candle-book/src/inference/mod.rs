#[cfg(test)]
mod tests {

    #[rustfmt::skip]
    #[tokio::test]
    async fn book_hub_1() {
// ANCHOR: book_hub_1_top
use candle::Device;
use hf_hub::api::tokio::Api;
// ANCHOR_END: book_hub_1_top
// ANCHOR: book_hub_1
    let api = Api::new().unwrap();
    let repo = api.model("bert-base-uncased".to_string());

    let weights_filename = repo.get("model.safetensors").await.unwrap();

    let weights = candle::safetensors::load(weights_filename, &Device::Cpu).unwrap();
    for (name, tensor) in weights.iter() {
        println!("Weight: {}, Dimensions: {:?}", name, tensor);
    }
// ANCHOR_END: book_hub_1
    assert_eq!(weights.len(), 206);
    }

}