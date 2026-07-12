import { SseDecoder } from './sse.js';
import { Transport } from './transport.js';
import {
  ChatMessage,
  ChatOptions,
  ChatResult,
  ClientOptions,
  ModelInfo,
  SapientHttpError,
} from './types.js';

const DEFAULT_BASE_URL = 'http://127.0.0.1:11435';

/**
 * Client for the SAPIENT engine, over a pluggable {@link Transport}.
 *
 * Default transport is HTTP to a running `sapient serve` (OpenAI-compatible)
 * — works anywhere `fetch` exists: Node.js ≥ 18, browsers, and React Native.
 * The first request for a model triggers its download + load
 * (Ollama-style lazy loading), so expect the first call to take a while.
 * Pass a `NativeTransport` (React Native on-device package) to run the
 * engine in-process instead — same API.
 */
export class SapientClient {
  private readonly transport: Transport;

  constructor(options: ClientOptions = {}) {
    this.transport = options.transport ?? new HttpTransport(options);
  }

  /** One chat turn, returned whole. */
  chat(messages: ChatMessage[], model: string, options: ChatOptions = {}): Promise<ChatResult> {
    return this.transport.chat(messages, model, options);
  }

  /**
   * One chat turn, streamed token-by-token as an async generator of text
   * fragments. Break out of the loop (or abort via `options.signal`) to
   * cancel generation.
   *
   * React Native's built-in fetch cannot stream response bodies — pass
   * `fetch` from `expo/fetch` in {@link ClientOptions}, or use `chat()`.
   */
  chatStream(
    messages: ChatMessage[],
    model: string,
    options: ChatOptions = {},
  ): AsyncGenerator<string, void, undefined> {
    return this.transport.chatStream(messages, model, options);
  }

  /** The models currently known to the engine. */
  models(signal?: AbortSignal): Promise<ModelInfo[]> {
    return this.transport.models(signal);
  }

  /** Liveness + resident-model report. */
  health(signal?: AbortSignal): Promise<Record<string, unknown>> {
    return this.transport.health(signal);
  }
}

/**
 * The default transport: OpenAI-compatible HTTP to `sapient serve`.
 * (This is the exact client behavior from before the Transport seam —
 * constructing `SapientClient` without a transport is unchanged.)
 */
export class HttpTransport implements Transport {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private readonly headers: Record<string, string>;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    // Wrap rather than store: bare `fetch` loses its `this` binding in
    // browsers/React Native when called as a method.
    const f = options.fetch ?? globalThis.fetch;
    if (!f) {
      throw new Error(
        'no fetch implementation available — use Node ≥ 18 or pass one via ClientOptions.fetch',
      );
    }
    this.fetchImpl = (...args: Parameters<typeof fetch>) => f(...args);
    this.headers = options.headers ?? {};
  }

  /** One chat turn, returned whole. */
  async chat(messages: ChatMessage[], model: string, options: ChatOptions = {}): Promise<ChatResult> {
    const res = await this.post(
      '/v1/chat/completions',
      { ...requestBody(messages, model, options), stream: false },
      options.signal,
    );
    const json = (await res.json()) as {
      model: string;
      choices: { message: { content: string }; finish_reason: string | null }[];
      usage?: ChatResult['usage'];
    };
    const choice = json.choices?.[0];
    return {
      content: choice?.message?.content ?? '',
      model: json.model,
      finishReason: choice?.finish_reason ?? null,
      usage: json.usage,
    };
  }

  /**
   * One chat turn, streamed token-by-token as an async generator of text
   * fragments. Break out of the loop (or abort via `options.signal`) to
   * cancel generation.
   *
   * React Native's built-in fetch cannot stream response bodies — pass
   * `fetch` from `expo/fetch` in {@link ClientOptions}, or use `chat()`.
   */
  async *chatStream(
    messages: ChatMessage[],
    model: string,
    options: ChatOptions = {},
  ): AsyncGenerator<string, void, undefined> {
    const res = await this.post(
      '/v1/chat/completions',
      { ...requestBody(messages, model, options), stream: true },
      options.signal,
    );
    if (!res.body) {
      throw new Error(
        'response body is not streamable in this environment — ' +
          'use chat() instead, or supply a streaming-capable fetch (e.g. expo/fetch)',
      );
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    const sse = new SseDecoder();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) return;
        for (const data of sse.push(decoder.decode(value, { stream: true }))) {
          if (data === '[DONE]') return;
          const chunk = JSON.parse(data) as {
            choices?: { delta?: { content?: string } }[];
          };
          const text = chunk.choices?.[0]?.delta?.content;
          if (text) yield text;
        }
      }
    } finally {
      // Break/early-return/abort: releasing the reader cancels the request,
      // which stops generation server-side.
      await reader.cancel().catch(() => {});
    }
  }

  /** The models currently known to the server. */
  async models(signal?: AbortSignal): Promise<ModelInfo[]> {
    const res = await this.get('/v1/models', signal);
    const json = (await res.json()) as { data?: ModelInfo[] };
    return json.data ?? [];
  }

  /** Liveness + resident-model report from `/v1/health`. */
  async health(signal?: AbortSignal): Promise<Record<string, unknown>> {
    const res = await this.get('/v1/health', signal);
    return (await res.json()) as Record<string, unknown>;
  }

  private async get(path: string, signal?: AbortSignal): Promise<Response> {
    const url = this.baseUrl + path;
    const res = await this.fetchImpl(url, { headers: this.headers, signal });
    return checkOk(res, url);
  }

  private async post(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
    const url = this.baseUrl + path;
    const res = await this.fetchImpl(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...this.headers },
      body: JSON.stringify(body),
      signal,
    });
    return checkOk(res, url);
  }
}

function requestBody(messages: ChatMessage[], model: string, options: ChatOptions) {
  return {
    model,
    messages,
    ...(options.maxTokens !== undefined && { max_tokens: options.maxTokens }),
    ...(options.temperature !== undefined && { temperature: options.temperature }),
    ...(options.topP !== undefined && { top_p: options.topP }),
    ...(options.stop !== undefined && { stop: options.stop }),
  };
}

async function checkOk(res: Response, url: string): Promise<Response> {
  if (res.ok) return res;
  const body = await res.text().catch(() => '');
  throw new SapientHttpError(res.status, body, url);
}
