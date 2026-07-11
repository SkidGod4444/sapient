// Integration tests against a real node:http server that mimics
// `sapient serve`'s OpenAI-compatible endpoints (shapes taken from
// crates/sapient-cli/src/server.rs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { SapientClient, SapientHttpError } from '../dist/index.js';

function chunk(text) {
  return (
    'data: ' +
    JSON.stringify({
      id: 'chatcmpl-1',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'test-model',
      choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
    }) +
    '\n\n'
  );
}

async function withServer(handler, run) {
  const server = createServer(handler);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const { port } = server.address();
  try {
    await run(new SapientClient({ baseUrl: `http://127.0.0.1:${port}` }), port);
  } finally {
    // A failed test can leave a streaming connection open; close() alone
    // would then hang until the runner's timeout.
    server.closeAllConnections?.();
    server.close();
  }
}

test('chat() returns the assistant reply and usage', async () => {
  await withServer(
    (req, res) => {
      assert.equal(req.url, '/v1/chat/completions');
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        const parsed = JSON.parse(body);
        assert.equal(parsed.stream, false);
        assert.equal(parsed.model, 'qwen2.5-0.5b');
        assert.equal(parsed.max_tokens, 64);
        assert.deepEqual(parsed.messages, [{ role: 'user', content: 'hi' }]);
        res.setHeader('content-type', 'application/json');
        res.end(
          JSON.stringify({
            id: 'chatcmpl-1',
            object: 'chat.completion',
            created: 0,
            model: 'qwen2.5-0.5b',
            choices: [
              { index: 0, message: { role: 'assistant', content: 'Hello!' }, finish_reason: 'stop' },
            ],
            usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
          }),
        );
      });
    },
    async (client) => {
      const out = await client.chat([{ role: 'user', content: 'hi' }], 'qwen2.5-0.5b', {
        maxTokens: 64,
      });
      assert.equal(out.content, 'Hello!');
      assert.equal(out.model, 'qwen2.5-0.5b');
      assert.equal(out.finishReason, 'stop');
      assert.equal(out.usage.total_tokens, 5);
    },
  );
});

test('chatStream() yields token fragments and stops at [DONE]', async () => {
  await withServer(
    (req, res) => {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        assert.equal(JSON.parse(body).stream, true);
        res.setHeader('content-type', 'text/event-stream');
        res.write(chunk('Hel'));
        res.write(chunk('lo'));
        res.write(chunk('!'));
        res.write('data: [DONE]\n\n');
        res.end();
      });
    },
    async (client) => {
      const got = [];
      for await (const t of client.chatStream([{ role: 'user', content: 'hi' }], 'test-model')) {
        got.push(t);
      }
      assert.deepEqual(got, ['Hel', 'lo', '!']);
    },
  );
});

test('breaking out of chatStream() cancels the request', async () => {
  let closed = false;
  await withServer(
    (req, res) => {
      req.on('data', () => {});
      req.on('end', () => {
        res.setHeader('content-type', 'text/event-stream');
        res.write(chunk('first'));
        // Keep the response open — only client cancellation ends it.
        res.on('close', () => {
          closed = true;
        });
      });
    },
    async (client) => {
      for await (const t of client.chatStream([{ role: 'user', content: 'hi' }], 'test-model')) {
        assert.equal(t, 'first');
        break; // cancel mid-stream
      }
      // The generator's finally block cancels the reader → socket closes.
      for (let i = 0; i < 50 && !closed; i++) await new Promise((r) => setTimeout(r, 20));
      assert.equal(closed, true);
    },
  );
});

test('models() unwraps the OpenAI list shape', async () => {
  await withServer(
    (req, res) => {
      assert.equal(req.url, '/v1/models');
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          object: 'list',
          data: [{ id: 'qwen2.5-0.5b', object: 'model', owned_by: 'openhorizon' }],
        }),
      );
    },
    async (client) => {
      const models = await client.models();
      assert.equal(models.length, 1);
      assert.equal(models[0].id, 'qwen2.5-0.5b');
    },
  );
});

test('health() returns the server report', async () => {
  await withServer(
    (req, res) => {
      assert.equal(req.url, '/v1/health');
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ status: 'ok', version: '0.5.3', resident_models: [] }));
    },
    async (client) => {
      const health = await client.health();
      assert.equal(health.status, 'ok');
      assert.deepEqual(health.resident_models, []);
    },
  );
});

test('non-2xx responses throw SapientHttpError with the body', async () => {
  await withServer(
    (req, res) => {
      res.statusCode = 404;
      res.end('model not found');
    },
    async (client) => {
      await assert.rejects(
        () => client.health(),
        (err) => {
          assert.ok(err instanceof SapientHttpError);
          assert.equal(err.status, 404);
          assert.equal(err.body, 'model not found');
          return true;
        },
      );
    },
  );
});
