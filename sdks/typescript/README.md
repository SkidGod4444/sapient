# @openhorizon/sapient

TypeScript SDK for [SAPIENT](https://github.com/SkidGod4444/sapient) — run
LLMs on your own hardware and talk to them from Node.js or React Native.

Today the SDK speaks to a local **`sapient serve`** instance over its
OpenAI-compatible HTTP API (streaming included). A native Node binding
(napi) and a React Native JSI module over `sapient-ffi` are the next rungs —
same API, no server process. See
[`docs/MOBILE.md`](../../docs/MOBILE.md) for the Phase-11 plan.

## Install & run

```bash
# 1. Start the engine (any machine on your network)
sapient serve            # listens on 127.0.0.1:11435

# 2. In your app
npm install @openhorizon/sapient
```

```ts
import { SapientClient } from '@openhorizon/sapient';

const client = new SapientClient(); // http://127.0.0.1:11435

// Whole reply at once
const { content } = await client.chat(
  [{ role: 'user', content: 'One-sentence fun fact about octopuses?' }],
  'qwen2.5-0.5b',
);

// Or streamed token-by-token
for await (const token of client.chatStream(
  [{ role: 'user', content: 'Tell me a haiku.' }],
  'qwen2.5-0.5b',
  { maxTokens: 64, temperature: 0.7 },
)) {
  process.stdout.write(token);
}
```

The first request for a model downloads + loads it server-side
(Ollama-style lazy loading) — expect the first call to take a while.

## React Native

Point the client at your dev machine and it works out of the box for
non-streamed calls:

```ts
const client = new SapientClient({ baseUrl: 'http://192.168.1.42:11435' });
```

RN's built-in `fetch` cannot stream response bodies. For `chatStream()`,
pass a streaming-capable fetch (e.g. `expo/fetch`):

```ts
import { fetch } from 'expo/fetch';
const client = new SapientClient({ baseUrl: 'http://192.168.1.42:11435', fetch });
```

This dev-loop keeps inference off your phone entirely while you build UI —
see the safe-testing guide in `docs/MOBILE.md`.

## API

- `new SapientClient({ baseUrl?, fetch?, headers? })`
- `chat(messages, model, { maxTokens?, temperature?, topP?, stop?, signal? })` → `{ content, model, finishReason, usage }`
- `chatStream(messages, model, options?)` → `AsyncGenerator<string>` — `break` or abort the `signal` to cancel generation
- `models()` → catalog/resident models (`/v1/models`)
- `health()` → liveness + resident-model report (`/v1/health`)
- Non-2xx responses throw `SapientHttpError` (`status`, `body`)

## Develop

```bash
npm install
npm test     # builds with tsc, then runs node --test (SSE units + a mock-serve integration suite)
```

Zero runtime dependencies; Node ≥ 18.
