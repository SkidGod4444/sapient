import { ChatMessage, ChatOptions, ChatResult, ModelInfo } from './types.js';

/**
 * The pluggable backend behind {@link SapientClient}. Two implementations:
 *
 * - `HttpTransport` (default, in this package) — talks to a running
 *   `sapient serve` over its OpenAI-compatible API. Works on Node ≥ 18,
 *   browsers, and React Native (pass `expo/fetch` for streaming).
 * - `NativeTransport` (ships with the React Native on-device package) —
 *   drives the engine in-process over `sapient-ffi`; no server involved.
 *
 * The client API is identical over both — UI code never changes when an app
 * graduates from the serve-backed dev loop to on-device inference.
 */
export interface Transport {
  chat(messages: ChatMessage[], model: string, options?: ChatOptions): Promise<ChatResult>;
  chatStream(
    messages: ChatMessage[],
    model: string,
    options?: ChatOptions,
  ): AsyncGenerator<string, void, undefined>;
  models(signal?: AbortSignal): Promise<ModelInfo[]>;
  health(signal?: AbortSignal): Promise<Record<string, unknown>>;
}
