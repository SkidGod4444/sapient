// On-device Transport for the SAPIENT TypeScript SDK (`@openhorizon/sapient`).
//
// Structurally implements the SDK's `Transport` interface — pass it via
// `new SapientClient({ transport: new NativeTransport() })` and existing UI
// code moves from the serve-backed dev loop to fully on-device inference
// with no other changes. The conversation stays CALLER-owned (the full
// message array is sent each turn, exactly like the HTTP transport); the
// engine's prefix cache makes the re-sent history cheap.
import {
  GenerationOptions,
  listModels,
  loadSession,
  setCacheDir,
  version,
  type LlmSessionInterface,
  type Message,
} from './generated/sapient_ffi';

/** Mirrors the SDK's wire types structurally (no runtime dependency). */
type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };
type ChatOptions = {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string | string[];
  signal?: AbortSignal;
};
type ChatResult = {
  content: string;
  model: string;
  finishReason: string | null;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
};
type ModelInfo = { id: string; object: string; owned_by?: string };

export interface NativeTransportOptions {
  /**
   * Sampling/config applied at model load (the engine fixes generation
   * config per loaded session; per-call ChatOptions.maxTokens etc. are
   * honored only if the model is not yet loaded).
   */
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  /** `auto` (default: GPU when available, CPU fallback) | `cpu` | `wgpu`. */
  backend?: string;
  /**
   * Model-cache directory (`HF_HOME`) — pass the app's caches dir (e.g.
   * `expo-file-system`'s `Paths.cache`/`cacheDirectory`, `file://` prefix
   * stripped) so the OS can reclaim downloads and uninstall removes them.
   * Applied before the first load.
   */
  cacheDir?: string;
}

/**
 * Runs the SAPIENT engine in-process over `sapient-ffi` (UniFFI → JSI).
 * One model resident at a time (phone RAM: see docs/MOBILE.md §5.2);
 * switching models drops the previous session.
 */
export class NativeTransport {
  private session: LlmSessionInterface | null = null;
  private loadedModel: string | null = null;
  private loading: Promise<LlmSessionInterface> | null = null;
  private readonly options: NativeTransportOptions;

  constructor(options: NativeTransportOptions = {}) {
    this.options = options;
  }

  /**
   * Load (download on first use) and hold a model resident. Called lazily by
   * chat/chatStream; call it eagerly to control when the download happens.
   */
  async loadModel(model: string, callOptions: ChatOptions = {}): Promise<void> {
    await this.ensureSession(model, callOptions);
  }

  /** The resolved engine backend of the loaded model (e.g. `wgpu (…)`). */
  backendLabel(): string | null {
    return this.session?.backendLabel() ?? null;
  }

  async chat(
    messages: ChatMessage[],
    model: string,
    options: ChatOptions = {},
  ): Promise<ChatResult> {
    let content = '';
    for await (const token of this.chatStream(messages, model, options)) {
      content += token;
    }
    return { content, model, finishReason: 'stop' };
  }

  async *chatStream(
    messages: ChatMessage[],
    model: string,
    options: ChatOptions = {},
  ): AsyncGenerator<string, void, undefined> {
    const session = await this.ensureSession(model, options);

    // Bridge the callback-push world onto an async-generator pull: tokens
    // queue between pulls; the generator's return path (break/abort) flips
    // `cancelled`, and the listener returning false halts the engine.
    const queue: string[] = [];
    let cancelled = false;
    let notify: (() => void) | null = null;
    let done = false;
    let failure: unknown = null;

    const turn = session
      .chatMessagesStream(messages as Message[], {
        onToken: (token: string): boolean => {
          queue.push(token);
          notify?.();
          return !cancelled;
        },
      })
      .catch((e: unknown) => {
        failure = e;
      })
      .finally(() => {
        done = true;
        notify?.();
      });

    const onAbort = () => {
      cancelled = true;
      notify?.();
    };
    options.signal?.addEventListener('abort', onAbort, { once: true });

    try {
      for (;;) {
        while (queue.length > 0) yield queue.shift() as string;
        if (done || cancelled) break;
        await new Promise<void>((resolve) => {
          notify = resolve;
        });
        notify = null;
      }
      // Drain whatever arrived between the last pull and completion.
      while (queue.length > 0 && !cancelled) yield queue.shift() as string;
      if (failure) throw failure;
    } finally {
      cancelled = true;
      options.signal?.removeEventListener('abort', onAbort);
      await turn; // the engine has stopped before the generator returns
    }
  }

  async models(): Promise<ModelInfo[]> {
    return listModels().map((m) => ({
      id: m.alias,
      object: 'model',
      owned_by: m.repoId,
    }));
  }

  async health(): Promise<Record<string, unknown>> {
    return {
      status: 'ok',
      mode: 'on-device',
      engine: version(),
      resident_models: this.loadedModel ? [this.loadedModel] : [],
      backend: this.backendLabel(),
    };
  }

  private async ensureSession(
    model: string,
    callOptions: ChatOptions,
  ): Promise<LlmSessionInterface> {
    if (this.session && this.loadedModel === model) return this.session;
    if (this.loading) await this.loading.catch(() => {});
    if (this.session && this.loadedModel === model) return this.session;

    if (this.options.cacheDir) setCacheDir(this.options.cacheDir);
    const opts = GenerationOptions.create({
      maxTokens: callOptions.maxTokens ?? this.options.maxTokens ?? 512,
      temperature: callOptions.temperature ?? this.options.temperature,
      topP: callOptions.topP ?? this.options.topP,
      backend: this.options.backend,
    });
    this.loading = loadSession(model, opts);
    try {
      this.session = await this.loading;
      this.loadedModel = model;
      return this.session;
    } finally {
      this.loading = null;
    }
  }
}
