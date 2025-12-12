use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn model_all_minilm_l6_v2() -> Result<(BertModel, Tokenizer)> {
    // Auto-detect best device
    let device = if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
        println!("Using CUDA GPU acceleration");
        Device::new_cuda(0)?
    } else if cfg!(feature = "metal") && candle_core::utils::metal_is_available() {
        println!("Using Metal GPU acceleration");
        Device::new_metal(0)?
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let revision = "main".to_string();

    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let api = Api::new()?;
    let api = api.repo(repo);

    // Download model files
    let config_filename = api.get("config.json")?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("model.safetensors")?;

    // Load config
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let config = Config {
        hidden_act: HiddenAct::GeluApproximate,
        ..config
    };

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // Load model weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
    };
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}


fn model_all_mpnet_base_v2() -> Result<(BertModel, Tokenizer)> {
    // Auto-detect best device
    let device = if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
        println!("Using CUDA GPU acceleration");
        Device::new_cuda(0)?
    } else if cfg!(feature = "metal") && candle_core::utils::metal_is_available() {
        println!("Using Metal GPU acceleration");
        Device::new_metal(0)?
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    let model_id = "sentence-transformers/all-mpnet-base-v2".to_string();
    let revision = "main".to_string();

    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let api = Api::new()?;
    let api = api.repo(repo);

    // Download model files
    let config_filename = api.get("config.json")?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let weights_filename = api.get("model.safetensors")?;

    // Load config
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let config = Config {
        hidden_act: HiddenAct::GeluApproximate,
        ..config
    };

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // Load model weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
    };
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}




fn get_embeddings(model: &BertModel, tokenizer: &Tokenizer, sentences: &[&str], device: &Device) -> Result<Tensor> {

    // Tokenize all sentences
    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg)?;

    // Get max length for padding
    let max_len = tokens.iter().map(|t| t.get_ids().len()).max().unwrap_or(0);

    // Prepare input tensors
    let mut token_ids_vec = Vec::new();
    let mut token_type_ids_vec = Vec::new();
    let mut attention_mask_vec = Vec::new();

    for encoding in &tokens {
        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        let attention = encoding.get_attention_mask();

        // Pad to max length
        let mut padded_ids: Vec<u32> = ids.to_vec();
        let mut padded_type_ids: Vec<u32> = type_ids.to_vec();
        let mut padded_attention: Vec<u32> = attention.to_vec();

        padded_ids.resize(max_len, 0);
        padded_type_ids.resize(max_len, 0);
        padded_attention.resize(max_len, 0);

        token_ids_vec.push(padded_ids);
        token_type_ids_vec.push(padded_type_ids);
        attention_mask_vec.push(padded_attention);
    }

    let batch_size = sentences.len();

    // Flatten and create tensors
    let token_ids: Vec<u32> = token_ids_vec.into_iter().flatten().collect();
    let token_type_ids: Vec<u32> = token_type_ids_vec.into_iter().flatten().collect();
    let attention_mask: Vec<u32> = attention_mask_vec.into_iter().flatten().collect();

    let token_ids = Tensor::from_vec(token_ids, (batch_size, max_len), device)?;
    let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, max_len), device)?;
    let attention_mask = Tensor::from_vec(attention_mask, (batch_size, max_len), device)?;

    // Run BERT forward pass
    let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

    // Mean pooling over the sequence dimension
    // embeddings shape: (batch_size, seq_len, hidden_size)
    let attention_mask = attention_mask.to_dtype(DType::F32)?;
    let attention_mask = attention_mask.unsqueeze(2)?; // (batch_size, seq_len, 1)

    let masked_embeddings = embeddings.broadcast_mul(&attention_mask)?;
    let sum_embeddings = masked_embeddings.sum(1)?; // (batch_size, hidden_size)
    let sum_mask = attention_mask.sum(1)?; // (batch_size, 1)

    let mean_embeddings = sum_embeddings.broadcast_div(&sum_mask)?;

    // L2 normalize
    let norm = mean_embeddings.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
    let normalized = mean_embeddings.broadcast_div(&norm)?;

    Ok(normalized)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    Ok(dot)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main() -> Result<()> {
        println!("Loading BERT model and tokenizer...");
        let (model, tokenizer) = model_all_mpnet_base_v2()?;
        let device = if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else if cfg!(feature = "metal") && candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };
        println!("Model loaded successfully!\n");

        // Example sentences for semantic similarity
        let sentences = [
            "The cat sits on the mat.",
            "A feline is resting on a rug.",
            "The weather is beautiful today.",
            "Machine learning is a subset of artificial intelligence.",
        ];

        println!("Computing embeddings for sentences:");
        for (i, s) in sentences.iter().enumerate() {
            println!("  {}: {}", i + 1, s);
        }
        println!();

        let embeddings = get_embeddings(&model, &tokenizer, &sentences, &device)?;

        // Compare sentence similarities
        println!("Cosine similarities:");
        println!("{}", "-".repeat(50));

        for i in 0..sentences.len() {
            for j in (i + 1)..sentences.len() {
                let emb_i = embeddings.get(i)?;
                let emb_j = embeddings.get(j)?;
                let similarity = cosine_similarity(&emb_i, &emb_j)?;
                println!(
                    "Sentences {} & {}: {:.4}",
                    i + 1,
                    j + 1,
                    similarity
                );
            }
        }

        println!("\nDone!");
        Ok(())
    }
}