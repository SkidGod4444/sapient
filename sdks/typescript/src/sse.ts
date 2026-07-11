/**
 * Incremental Server-Sent-Events decoder.
 *
 * Feed it raw text chunks as they arrive off the wire; it returns the
 * `data:` payload of every event that the chunk completes. Handles events
 * split across chunk boundaries, multi-`data:`-line events (joined with
 * `\n` per the SSE spec), `\r\n` line endings, and ignores comments and
 * non-`data` fields — which is all `sapient serve`'s OpenAI-style stream
 * needs.
 */
export class SseDecoder {
  private buffer = '';

  /** Feed a chunk; returns the data payloads of any events it completed. */
  push(chunk: string): string[] {
    this.buffer += chunk;
    const events: string[] = [];
    for (;;) {
      const match = this.buffer.match(/\r?\n\r?\n/);
      if (!match || match.index === undefined) break;
      const raw = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const data = SseDecoder.dataOf(raw);
      if (data !== null) events.push(data);
    }
    return events;
  }

  /** Extract the joined `data:` payload of one raw event block, or null. */
  private static dataOf(raw: string): string | null {
    const lines: string[] = [];
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith('data:')) {
        // Per spec, a single leading space after the colon is stripped.
        lines.push(line.slice(5).replace(/^ /, ''));
      }
    }
    return lines.length > 0 ? lines.join('\n') : null;
  }
}
