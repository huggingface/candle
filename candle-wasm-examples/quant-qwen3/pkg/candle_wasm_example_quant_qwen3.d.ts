/* tslint:disable */
/* eslint-disable */

/**
 * A chat message with role and content
 */
export class Message {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Create a new message with the given role and content
     */
    constructor(role: string, content: string);
    /**
     * Get the message content
     */
    readonly content: string;
    /**
     * Get the message role
     */
    readonly role: string;
}

export class Model {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Send a user message and prepare for generation.
     *
     * This method efficiently reuses the KV cache by only tokenizing NEW content:
     * - First turn: tokenizes full prompt (system + user + assistant start)
     * - Subsequent turns: tokenizes only the continuation (close prev + new user + assistant start)
     *
     * The `enable_thinking` parameter controls whether this specific message should use thinking mode.
     */
    chat(user_message: string, temp: number, top_p: number, repeat_penalty: number, repeat_last_n: number, seed: number, enable_thinking: boolean): string;
    /**
     * Clear conversation history but keep system prompt.
     * Also clears KV cache since we're starting fresh.
     */
    clear_conversation(): void;
    /**
     * Complete the current turn and record the assistant response.
     * The generated tokens remain in the KV cache for the next turn.
     */
    end_turn(): void;
    /**
     * Generate multiple tokens at once.
     */
    generate_tokens(count: number): string;
    /**
     * Get number of tokens currently in KV cache.
     */
    get_cached_token_count(): number;
    /**
     * Get conversation history as JSON.
     */
    get_conversation_json(): string;
    /**
     * Get number of messages in conversation.
     */
    get_message_count(): number;
    /**
     * Get total token count in KV cache.
     */
    get_token_count(): number;
    /**
     * Check if the last generated token was EOS.
     */
    is_eos(): boolean;
    constructor(weights: Uint8Array, tokenizer: Uint8Array, _config: Uint8Array);
    /**
     * Generate the next token.
     */
    next_token(): string;
    /**
     * Reset the model completely (clears KV cache and all state).
     */
    reset(): void;
    /**
     * Initialize a new conversation with system prompt and options.
     * This clears the KV cache and starts fresh.
     */
    start_conversation(system_prompt: string | null | undefined, enable_thinking: boolean): void;
    /**
     * Load conversation template from tokenizer_config.json content.
     */
    start_conversation_from_config(tokenizer_config_json: string, system_prompt: string | null | undefined, enable_thinking: boolean): void;
}

export class ProfileStats {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly json: string;
}

export function get_memory_info(): string;

export function get_wasm_memory_info(): string;

export function initThreadPool(num_threads: number): Promise<any>;

export function log_memory(): void;

export function log_wasm_memory(): void;

export function profile_clear(): void;

export function profile_enable(enabled: boolean): void;

export function profile_get_stats(): ProfileStats;

export function profile_print_stats(): void;

export class wbg_rayon_PoolBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    build(): void;
    numThreads(): number;
    receiver(): number;
}

export function wbg_rayon_start_worker(receiver: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly __wbg_model_free: (a: number, b: number) => void;
    readonly model_chat: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly model_clear_conversation: (a: number) => void;
    readonly model_end_turn: (a: number) => void;
    readonly model_generate_tokens: (a: number, b: number) => [number, number, number, number];
    readonly model_get_cached_token_count: (a: number) => number;
    readonly model_get_conversation_json: (a: number) => [number, number];
    readonly model_get_message_count: (a: number) => number;
    readonly model_is_eos: (a: number) => number;
    readonly model_load: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly model_next_token: (a: number) => [number, number, number, number];
    readonly model_reset: (a: number) => void;
    readonly model_start_conversation: (a: number, b: number, c: number, d: number) => void;
    readonly model_start_conversation_from_config: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly model_get_token_count: (a: number) => number;
    readonly __wbg_profilestats_free: (a: number, b: number) => void;
    readonly get_memory_info: () => [number, number];
    readonly get_wasm_memory_info: () => [number, number];
    readonly log_memory: () => void;
    readonly log_wasm_memory: () => void;
    readonly profile_enable: (a: number) => void;
    readonly profile_get_stats: () => number;
    readonly profile_print_stats: () => void;
    readonly profilestats_json: (a: number) => [number, number];
    readonly profile_clear: () => void;
    readonly __wbg_wbg_rayon_poolbuilder_free: (a: number, b: number) => void;
    readonly initThreadPool: (a: number) => any;
    readonly wbg_rayon_poolbuilder_build: (a: number) => void;
    readonly wbg_rayon_poolbuilder_numThreads: (a: number) => number;
    readonly wbg_rayon_poolbuilder_receiver: (a: number) => number;
    readonly wbg_rayon_start_worker: (a: number) => void;
    readonly __wbg_message_free: (a: number, b: number) => void;
    readonly message_content: (a: number) => [number, number];
    readonly message_new: (a: number, b: number, c: number, d: number) => number;
    readonly message_role: (a: number) => [number, number];
    readonly memory: WebAssembly.Memory;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_thread_destroy: (a?: number, b?: number, c?: number) => void;
    readonly __wbindgen_start: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number }} module - Passing `SyncInitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number } | SyncInitInput, memory?: WebAssembly.Memory): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number }} module_or_path - Passing `InitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number } | InitInput | Promise<InitInput>, memory?: WebAssembly.Memory): Promise<InitOutput>;
