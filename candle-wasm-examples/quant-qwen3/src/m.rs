use candle::quantized::gguf_file;
use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_wasm_chat_template::{ChatTemplate, ChatTemplateOptions, Conversation, Message};
use js_sys::Date;
use std::io::Cursor;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

use crate::console_log;
use crate::profiler::ProfileGuard;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;

#[wasm_bindgen]
pub struct Model {
    model: QuantizedQwen3,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
    enable_thinking: bool,

    // === KV Cache Management ===
    /// Actual token IDs that are in the KV cache.
    /// This is the source of truth for what's been processed.
    kv_tokens: Vec<u32>,

    /// Tokens generated during the current assistant turn.
    current_gen_tokens: Vec<u32>,

    // === Conversation State ===
    /// Text-level conversation history (for export/display).
    conversation: Option<Conversation>,

    /// Accumulator for current assistant response text during generation.
    current_response: String,

    /// Track whether this is the first turn (need full template) or continuation.
    is_first_turn: bool,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        _config: Vec<u8>,
    ) -> Result<Model, JsError> {
        let _prof = ProfileGuard::new("total_load");
        console_error_panic_hook::set_once();

        let device = Device::Cpu;

        let _prof = ProfileGuard::new("load_tokenizer");
        console_log!("Loading tokenizer...");
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;

        // Get EOS token
        let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(&token) => token,
            None => match tokenizer.get_vocab(true).get("<|im_end|>") {
                Some(&token) => token,
                None => {
                    console_log!("Warning: no EOS token found, using 0");
                    0
                }
            },
        };

        let start = Date::now();
        console_log!(
            "Weights size: {} bytes ({:.2} MB)",
            weights.len(),
            weights.len() as f64 / 1_048_576.0
        );

        let model = {
            let _prof = ProfileGuard::new("parse_gguf");

            let mut cursor = Cursor::new(weights);
            let content = gguf_file::Content::read(&mut cursor)
                .map_err(|e| JsError::new(&format!("Failed to read GGUF: {}", e)))?;

            console_log!("GGUF file parsed, loading model weights...");

            QuantizedQwen3::from_gguf(content, &mut cursor, &device)?
        };

        let load_time = (Date::now() - start) / 1000.0;
        console_log!("Quantized model loaded in {:.2}s", load_time);

        let logits_processor = LogitsProcessor::new(299792458, None, None);

        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            eos_token,
            enable_thinking: true,
            kv_tokens: Vec::new(),
            current_gen_tokens: Vec::new(),
            conversation: None,
            current_response: String::new(),
            is_first_turn: true,
        })
    }

    // ========================================================================
    // Conversation Management
    // ========================================================================

    /// Initialize a new conversation with system prompt and options.
    /// This clears the KV cache and starts fresh.
    #[wasm_bindgen]
    pub fn start_conversation(&mut self, system_prompt: Option<String>, enable_thinking: bool) {
        let _prof = ProfileGuard::new("start_conversation");

        self.enable_thinking = enable_thinking;

        // Clear KV cache for new conversation
        self.model.clear_kv_cache();
        self.kv_tokens.clear();
        self.current_gen_tokens.clear();
        self.current_response.clear();
        self.is_first_turn = true;

        // Build proper system prompt with metadata
        let reasoning_mode = if enable_thinking { "/think" } else { "/no_think" };
        let default_system = format!(
            "## Metadata\n\n\
Reasoning Mode: {}\n\n\
## Custom Instructions\n\n\
You are a helpful AI assistant.",
            reasoning_mode
        );

        let system = system_prompt.unwrap_or(default_system);

        let template = ChatTemplate::chatml_with_thinking();
        let options = ChatTemplateOptions::for_generation().thinking(enable_thinking);
        let conv = Conversation::new(template, system).with_options(options);

        self.conversation = Some(conv);

        console_log!(
            "Conversation started (reasoning mode: {})",
            reasoning_mode
        );
    }

    /// Load conversation template from tokenizer_config.json content.
    #[wasm_bindgen]
    pub fn start_conversation_from_config(
        &mut self,
        tokenizer_config_json: &str,
        system_prompt: Option<String>,
        enable_thinking: bool,
    ) -> Result<(), JsError> {
        let _prof = ProfileGuard::new("start_conversation_from_config");

        self.enable_thinking = enable_thinking;

        // Clear KV cache for new conversation
        self.model.clear_kv_cache();
        self.kv_tokens.clear();
        self.current_gen_tokens.clear();
        self.current_response.clear();
        self.is_first_turn = true;

        let template = ChatTemplate::from_config_json(tokenizer_config_json)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let options = ChatTemplateOptions::for_generation().thinking(enable_thinking);

        let conv = match system_prompt {
            Some(prompt) => Conversation::new(template, prompt).with_options(options),
            None => Conversation::without_system(template).with_options(options),
        };

        self.conversation = Some(conv);

        console_log!("Conversation started from config");
        Ok(())
    }

    /// Send a user message and prepare for generation.
    ///
    /// This method efficiently reuses the KV cache by only tokenizing NEW content:
    /// - First turn: tokenizes full prompt (system + user + assistant start)
    /// - Subsequent turns: tokenizes only the continuation (close prev + new user + assistant start)
    ///
    /// The `enable_thinking` parameter controls whether this specific message should use thinking mode.
    #[wasm_bindgen]
    pub fn chat(
        &mut self,
        user_message: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: f64,
        enable_thinking: bool,
    ) -> Result<String, JsError> {
        let _prof = ProfileGuard::new("chat");

        // Ensure conversation exists
        if self.conversation.is_none() {
            self.start_conversation(None, enable_thinking);
        }

        // Update thinking mode for this message
        self.enable_thinking = enable_thinking;

        // Clear generation state for new turn
        self.current_gen_tokens.clear();
        self.current_response.clear();

        // Setup logits processor
        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };
        self.logits_processor = LogitsProcessor::new(seed as u64, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;

        // Tokenize ONLY new content (not the full conversation)
        let new_tokens = if self.is_first_turn {
            let conv = self.conversation.as_mut()
                .ok_or_else(|| JsError::new("No conversation initialized"))?;

            // Update thinking mode for this specific turn
            conv.set_options(ChatTemplateOptions::for_generation().thinking(enable_thinking));

            // user_turn() adds the message AND returns the formatted prompt
            let prompt = conv.user_turn(&user_message)
                .map_err(|e| JsError::new(&e.to_string()))?;

            console_log!("First turn prompt:\n{}", prompt);

            let tokens = {
                let _prof = ProfileGuard::new("tokenize_prompt");
                self.tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(|m| JsError::new(&m.to_string()))?
                    .get_ids()
                    .to_vec()
            };

            self.is_first_turn = false;
            tokens
        } else {
            // Subsequent turns: only tokenize the continuation
            // Add to conversation history (for text export)
            if let Some(conv) = self.conversation.as_mut() {
                conv.add_message(Message::user(&user_message));
            }

            // Format only the new part: close previous assistant + new user + assistant start
            let continuation = self.format_continuation(&user_message, enable_thinking);

            let tokens = {
                let _prof = ProfileGuard::new("tokenize_continuation");
                self.tokenizer
                    .encode(continuation.as_str(), false) // false = don't add special tokens
                    .map_err(|m| JsError::new(&m.to_string()))?
                    .get_ids()
                    .to_vec()
            };

            tokens
        };

        let start_pos = self.kv_tokens.len();
        let num_messages = self.conversation.as_ref().map(|c| c.len()).unwrap_or(0);

        console_log!(
            "Chat: {} messages, {} cached tokens, {} new tokens, thinking: {}",
            num_messages,
            start_pos,
            new_tokens.len(),
            if enable_thinking { "on" } else { "off" }
        );

        if new_tokens.is_empty() {
            return Ok(String::new());
        }

        // Process new tokens and get first generated token
        let (text, first_gen_token) = self
            .process_prompt(&new_tokens, start_pos)
            .map_err(|m| JsError::new(&m.to_string()))?;

        // Update KV token tracking: only add prompt tokens (they're now in KV cache)
        // The first_gen_token is NOT in KV cache yet - it will be processed in next_token()
        self.kv_tokens.extend_from_slice(&new_tokens);
        self.current_gen_tokens.push(first_gen_token);

        // Accumulate response
        self.current_response.push_str(&text);

        Ok(text)
    }

    /// Complete the current turn and record the assistant response.
    /// The generated tokens remain in the KV cache for the next turn.
    #[wasm_bindgen]
    pub fn end_turn(&mut self) {
        let _prof = ProfileGuard::new("end_turn");

        if let Some(conv) = self.conversation.as_mut() {
            // Record the full response text in conversation history
            let response = self.current_response.clone();
            conv.assistant_response(&response);

            // Note: current_gen_tokens contains all generated tokens, but only len-1 are in KV cache
            // (the last one hasn't been processed yet, but it's EOS so that's fine)
            console_log!(
                "Turn ended: {} messages, {} tokens in KV cache, {} tokens generated",
                conv.len(),
                self.kv_tokens.len(),
                self.current_gen_tokens.len()
            );
        }

        self.current_response.clear();
        self.current_gen_tokens.clear();
    }

    /// Clear conversation history but keep system prompt.
    /// Also clears KV cache since we're starting fresh.
    #[wasm_bindgen]
    pub fn clear_conversation(&mut self) {
        if let Some(conv) = self.conversation.as_mut() {
            conv.clear();
        }
        self.model.clear_kv_cache();
        self.kv_tokens.clear();
        self.current_gen_tokens.clear();
        self.current_response.clear();
        self.is_first_turn = true;
        console_log!("Conversation cleared");
    }

    /// Get conversation history as JSON.
    #[wasm_bindgen]
    pub fn get_conversation_json(&self) -> String {
        match &self.conversation {
            Some(conv) => conv.to_json(),
            None => "[]".to_string(),
        }
    }

    /// Get number of messages in conversation.
    #[wasm_bindgen]
    pub fn get_message_count(&self) -> usize {
        match &self.conversation {
            Some(conv) => conv.len(),
            None => 0,
        }
    }

    /// Get number of tokens currently in KV cache.
    #[wasm_bindgen]
    pub fn get_cached_token_count(&self) -> usize {
        self.kv_tokens.len()
    }

    // ========================================================================
    // Token Generation
    // ========================================================================

    /// Generate the next token.
    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let _prof = ProfileGuard::new("next_token");

        // Get the last sampled token (which hasn't been processed/added to KV yet)
        let token_to_process = *self
            .current_gen_tokens
            .last()
            .ok_or_else(|| JsError::new("No tokens to continue from"))?;

        let text = self
            .process_generation(token_to_process)
            .map_err(|m| JsError::new(&m.to_string()))?;

        // Accumulate response
        self.current_response.push_str(&text);

        Ok(text)
    }

    /// Check if the last generated token was EOS.
    #[wasm_bindgen]
    pub fn is_eos(&self) -> bool {
        self.current_gen_tokens
            .last()
            .map_or(false, |&t| t == self.eos_token)
    }

    /// Get total token count in KV cache.
    #[wasm_bindgen]
    pub fn get_token_count(&self) -> usize {
        self.kv_tokens.len()
    }

    /// Generate multiple tokens at once.
    #[wasm_bindgen]
    pub fn generate_tokens(&mut self, count: usize) -> Result<String, JsError> {
        let _prof = ProfileGuard::new("generate_tokens_batch");

        let mut result = String::new();

        for _ in 0..count {
            if self.is_eos() {
                break;
            }

            let text = self.next_token()?;
            result.push_str(&text);
        }

        Ok(result)
    }

    /// Reset the model completely (clears KV cache and all state).
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        let _prof = ProfileGuard::new("reset_model");
        self.kv_tokens.clear();
        self.current_gen_tokens.clear();
        self.current_response.clear();
        self.conversation = None;
        self.is_first_turn = true;
        self.model.clear_kv_cache();
    }

}

// ============================================================================
// Private Implementation
// ============================================================================

impl Model {
    /// Format the continuation for a subsequent turn.
    /// This only generates the tokens needed to: close previous turn, add user message, start assistant.
    /// The KV cache already has everything before this.
    fn format_continuation(&self, user_message: &str, enable_thinking: bool) -> String {
        // ChatML format continuation:
        // <|im_end|>           (close previous assistant turn)
        // <|im_start|>user
        // {user_message}<|im_end|>
        // <|im_start|>assistant
        // <think>              (always present)
        // \n</think>\n         (pre-filled if no_think mode to skip reasoning)
        //
        // Note: Reasoning mode is set in system prompt at conversation start,
        // but we can still guide per-message behavior with think tag pre-filling

        let assistant_start = if enable_thinking {
            "<|im_start|>assistant\n<think>\n" // Open for reasoning
        } else {
            "<|im_start|>assistant\n<think>\n\n</think>\n" // Empty = skip reasoning
        };

        let result = format!(
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n{}",
            user_message,
            assistant_start
        );

        console_log!("Continuation format:\n{}", result);
        result
    }

    /// Process prompt tokens and return the first generated token.
    /// Note: This updates KV cache internally but does NOT modify kv_tokens.
    /// The caller (chat/init_with_prompt) is responsible for token tracking.
    fn process_prompt(&mut self, tokens: &[u32], start_pos: usize) -> candle::Result<(String, u32)> {
        let _prof = ProfileGuard::new("process_prompt");

        let dev = Device::Cpu;

        let input = {
            let _prof = ProfileGuard::new("create_input_tensor");
            Tensor::new(tokens, &dev)?.unsqueeze(0)?
        };

        // Forward pass through all prompt tokens
        let logits = {
            let _prof = ProfileGuard::new("model_forward_prompt");
            self.model.forward(&input, start_pos)?
        };

        let logits = {
            let _prof = ProfileGuard::new("logits_post_process");
            logits.squeeze(0)?.to_dtype(DType::F32)?
        };

        // Apply repeat penalty using all tokens (cached + new prompt tokens)
        let all_context: Vec<u32> = self
            .kv_tokens
            .iter()
            .chain(tokens.iter())
            .copied()
            .collect();

        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let _prof = ProfileGuard::new("apply_repeat_penalty");
            let start_at = all_context.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &all_context[start_at..],
            )?
        };

        // Sample first token
        let next_token = {
            let _prof = ProfileGuard::new("sample_token");
            self.logits_processor.sample(&logits)?
        };

        // Decode token
        let token_str = {
            let _prof = ProfileGuard::new("decode_token");
            match self.tokenizer.decode(&[next_token], false) {
                Ok(s) => s,
                Err(e) => {
                    console_log!("Error decoding token: {:?}", e);
                    String::new()
                }
            }
        };

        Ok((token_str, next_token))
    }

    /// Process a single token during generation.
    /// The token passed in is NOT yet in kv_tokens - it will be added after processing.
    fn process_generation(&mut self, token_to_process: u32) -> candle::Result<String> {
        let _prof = ProfileGuard::new("process_generation");

        let dev = Device::Cpu;

        let input = {
            let _prof = ProfileGuard::new("create_input_tensor");
            Tensor::new(&[token_to_process], &dev)?.unsqueeze(0)?
        };

        // Position is the next slot in the sequence (token_to_process hasn't been added yet)
        let pos = self.kv_tokens.len();

        // Forward pass for single token - this adds it to KV cache
        let logits = {
            let _prof = ProfileGuard::new("model_forward_gen");
            self.model.forward(&input, pos)?
        };

        // NOW add the processed token to kv_tokens (it's in KV cache now)
        self.kv_tokens.push(token_to_process);

        let logits = {
            let _prof = ProfileGuard::new("logits_post_process");
            logits.squeeze(0)?.to_dtype(DType::F32)?
        };

        // Apply repeat penalty
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let _prof = ProfileGuard::new("apply_repeat_penalty");
            let start_at = self.kv_tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &self.kv_tokens[start_at..],
            )?
        };

        // Sample next token
        let next_token = {
            let _prof = ProfileGuard::new("sample_token");
            self.logits_processor.sample(&logits)?
        };

        // Track the newly sampled token (NOT in kv_tokens yet - will be processed next iteration)
        self.current_gen_tokens.push(next_token);

        // Decode token
        let token_str = {
            let _prof = ProfileGuard::new("decode_token");
            match self.tokenizer.decode(&[next_token], false) {
                Ok(s) => s,
                Err(e) => {
                    console_log!("Error decoding token: {:?}", e);
                    String::new()
                }
            }
        };

        Ok(token_str)
    }
}
