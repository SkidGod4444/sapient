/** Message roles accepted by the chat endpoint. */
export type ChatRole = 'system' | 'user' | 'assistant';

/** One turn of a conversation. */
export interface ChatMessage {
  role: ChatRole;
  content: string;
}

/** Per-request generation options (OpenAI-compatible field semantics). */
export interface ChatOptions {
  /** Hard cap on new tokens for this reply. */
  maxTokens?: number;
  /** Sampling temperature; omit for the server's default. */
  temperature?: number;
  /** Nucleus sampling threshold (0–1). */
  topP?: number;
  /** Stop strings — generation ends before any of these appear. */
  stop?: string[];
  /** Abort the request (also cancels an in-flight stream). */
  signal?: AbortSignal;
}

/** Token accounting as reported by the server. */
export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** A completed (non-streamed) chat turn. */
export interface ChatResult {
  /** The assistant reply text. */
  content: string;
  /** The model that served the request. */
  model: string;
  /** Why generation stopped (`stop`, `length`, …). */
  finishReason: string | null;
  usage?: Usage;
}

/** One row of `GET /v1/models`. */
export interface ModelInfo {
  id: string;
  object: string;
  owned_by?: string;
}

/** Constructor options for {@link SapientClient}. */
export interface ClientOptions {
  /**
   * Base URL of the `sapient serve` instance. Defaults to
   * `http://127.0.0.1:11435` (the serve default port). For React Native
   * development point this at your dev machine's LAN IP.
   */
  baseUrl?: string;
  /**
   * Custom fetch implementation. Defaults to `globalThis.fetch` (Node 18+,
   * browsers, React Native). RN's built-in fetch cannot stream response
   * bodies — pass `fetch` from `expo/fetch` (or use `chat()` instead of
   * `chatStream()`).
   */
  fetch?: typeof fetch;
  /** Extra headers sent with every request (e.g. auth in front of a proxy). */
  headers?: Record<string, string>;
}

/** Error thrown for non-2xx responses, carrying the response body. */
export class SapientHttpError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string, url: string) {
    super(`sapient serve request failed: ${status} ${url}${body ? ` — ${body}` : ''}`);
    this.name = 'SapientHttpError';
    this.status = status;
    this.body = body;
  }
}
