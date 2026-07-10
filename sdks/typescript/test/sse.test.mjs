import test from 'node:test';
import assert from 'node:assert/strict';
import { SseDecoder } from '../dist/sse.js';

test('decodes a complete event', () => {
  const d = new SseDecoder();
  assert.deepEqual(d.push('data: {"a":1}\n\n'), ['{"a":1}']);
});

test('handles events split across chunk boundaries', () => {
  const d = new SseDecoder();
  assert.deepEqual(d.push('data: {"a"'), []);
  assert.deepEqual(d.push(':1}\n'), []);
  assert.deepEqual(d.push('\ndata: [DONE]\n\n'), ['{"a":1}', '[DONE]']);
});

test('joins multi-data-line events with newline', () => {
  const d = new SseDecoder();
  assert.deepEqual(d.push('data: line1\ndata: line2\n\n'), ['line1\nline2']);
});

test('tolerates CRLF line endings', () => {
  const d = new SseDecoder();
  assert.deepEqual(d.push('data: x\r\n\r\ndata: y\r\n\r\n'), ['x', 'y']);
});

test('ignores comments and non-data fields', () => {
  const d = new SseDecoder();
  assert.deepEqual(d.push(': keep-alive\n\nevent: ping\nid: 3\n\ndata: z\n\n'), ['z']);
});

test('strips only a single leading space after the colon', () => {
  const d = new SseDecoder();
  assert.deepEqual(d.push('data:  two spaces\n\ndata:none\n\n'), [' two spaces', 'none']);
});
