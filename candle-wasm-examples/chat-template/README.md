# candle-wasm-chat-template

Shared chat template support for candle WASM LLM examples.

## Features

- **Jinja templates**: Full MiniJinja support for HuggingFace-compatible templates
- **Preset templates**: Built-in support for ChatML, Llama 2/3, Mistral, Gemma, Phi-3
- **Multi-turn conversations**: `Conversation` manager handles history
- **Thinking mode**: Support for reasoning models (SmolLM3, Qwen3, DeepSeek)
- **WASM-ready**: Works in browser via wasm-bindgen

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
candle-wasm-chat-template = { path = "../chat-template" }
```

### Rust

```rust
use candle_wasm_chat_template::{ChatTemplate, Message, Conversation, ChatTemplateOptions};

// Use a preset template
let template = ChatTemplate::chatml_with_thinking();

// Single-turn
let messages = vec![
    Message::system("You are helpful."),
    Message::user("Hello!"),
];
let prompt = template.apply(&messages, &ChatTemplateOptions::for_generation())?;

// Multi-turn conversation
let mut conv = Conversation::new(ChatTemplate::chatml(), "You are helpful.");
let prompt1 = conv.user_turn("Hello!")?;
// ... generate response ...
conv.assistant_response("Hi there!");
let prompt2 = conv.user_turn("How are you?")?; // includes full history
```

### JavaScript (WASM)

```javascript
// Start conversation
model.start_conversation("You are helpful.", false);  // system prompt, thinking

// Chat turn
model.chat("Hello!", 0.7, 0.9, 1.1, 64, 12345);
while (!model.is_eos()) {
    output += model.next_token();
}
model.end_turn();

// Second turn includes history
model.chat("Follow up question...", 0.7, 0.9, 1.1, 64, 12345);
// ...
```

## Supported Templates

| Template | Models | Method |
|----------|--------|--------|
| ChatML | SmolLM, Qwen, many others | `ChatTemplate::chatml()` |
| ChatML + Thinking | SmolLM3, Qwen3 | `ChatTemplate::chatml_with_thinking()` |
| Llama 2 | Llama 2 Chat | `ChatTemplate::llama2()` |
| Llama 3 | Llama 3, 3.1, 3.2 | `ChatTemplate::llama3()` |
| Mistral | Mistral Instruct | `ChatTemplate::mistral()` |
| Gemma | Gemma, Gemma 2 | `ChatTemplate::gemma()` |
| Phi-3 | Phi-3 | `ChatTemplate::phi3()` |
| Custom | Any | `ChatTemplate::from_config_json(json)` |

## Loading from tokenizer_config.json

```rust
// In WASM, pass the JSON string from JavaScript
let template = ChatTemplate::from_config_json(tokenizer_config_json)?;
```