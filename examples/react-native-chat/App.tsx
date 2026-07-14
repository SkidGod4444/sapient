// SAPIENT chat over the TypeScript SDK — now with BOTH transports:
//
//   • on-device (default): the engine runs inside the app over
//     @openhorizon-labs/sapient-react-native (sapient-ffi → UniFFI → JSI).
//     Needs a development build (`npx expo prebuild` + run) — Expo Go
//     cannot load native code.
//   • server: HTTP to `sapient serve` (the rung-0 dev loop from
//     docs/MOBILE.md; streaming via expo/fetch because RN's built-in
//     fetch cannot stream response bodies).
//
// The SapientClient API is identical over both — only the transport
// changes. Toggle at runtime in the header.
import { SapientClient, type ChatMessage } from '@openhorizon-labs/sapient';
import { NativeTransport } from '@openhorizon-labs/sapient-react-native';
import { fetch as expoFetch } from 'expo/fetch';
import { Paths } from 'expo-file-system';
import { StatusBar } from 'expo-status-bar';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Button,
  FlatList,
  KeyboardAvoidingView,
  Linking,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

interface Bubble {
  id: string;
  role: 'user' | 'assistant';
  text: string;
}

type Status =
  | { kind: 'idle' }
  | { kind: 'generating' }
  | { kind: 'error'; detail: string };

type Mode = 'device' | 'server';

// Keep model downloads inside the app sandbox's Caches so the OS can
// reclaim them and uninstall removes them (docs/MOBILE.md §5.4).
// The engine wants a plain filesystem path, so strip expo-file-system's
// `file://` URI scheme. (SDK 54 replaced `FileSystem.cacheDirectory` with
// the `Paths` API.)
const cacheDir = Paths.cache?.uri
  ? Paths.cache.uri.replace(/^file:\/\//, '').replace(/\/?$/, '/') + 'sapient'
  : undefined;

export default function App() {
  const [mode, setMode] = useState<Mode>('device');
  // Simulators/emulators can reach the host via localhost; a physical phone
  // needs the dev machine's LAN IP (same Wi-Fi). Android emulator: 10.0.2.2.
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:11435');
  // Dev default per docs/MOBILE.md §5.2 — smallest model until boring.
  const [model, setModel] = useState('smollm2-135m-q4');
  const [draft, setDraft] = useState('');
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const [backend, setBackend] = useState<string | null>(null);
  const history = useRef<ChatMessage[]>([]);
  const abort = useRef<AbortController | null>(null);
  // Bumped by every send and by Clear — stale writes from an old stream
  // (status flips, history pushes) compare against it and drop.
  const turn = useRef(0);
  const listRef = useRef<FlatList<Bubble>>(null);

  // One resident model at a time — the transport survives mode flips so a
  // loaded model stays warm when toggling back.
  const native = useMemo(() => new NativeTransport({ cacheDir }), []);
  const client = useMemo(
    () =>
      mode === 'device'
        ? new SapientClient({ transport: native })
        : new SapientClient({ baseUrl, fetch: expoFetch as unknown as typeof fetch }),
    [mode, baseUrl, native],
  );

  const send = useCallback(async (promptOverride?: string) => {
    const prompt = (promptOverride ?? draft).trim();
    if (!prompt || status.kind === 'generating') return;
    setDraft('');

    const replyId = `${Date.now()}-a`;
    setBubbles((prev) => [
      ...prev,
      { id: `${Date.now()}-u`, role: 'user', text: prompt },
      { id: replyId, role: 'assistant', text: '' },
    ]);
    setStatus({ kind: 'generating' });

    history.current.push({ role: 'user', content: prompt });
    const myTurn = ++turn.current;
    const controller = new AbortController();
    abort.current = controller;
    let reply = '';
    try {
      for await (const token of client.chatStream(history.current, model, {
        maxTokens: 512,
        signal: controller.signal,
      })) {
        reply += token;
        setBubbles((prev) =>
          prev.map((b) => (b.id === replyId ? { ...b, text: reply } : b)),
        );
      }
      if (myTurn !== turn.current) return; // Clear reset history under us
      history.current.push({ role: 'assistant', content: reply });
      setStatus({ kind: 'idle' });
      if (mode === 'device') setBackend(native.backendLabel());
    } catch (e) {
      if (myTurn !== turn.current) return; // Clear reset history under us
      if (controller.signal.aborted) {
        // Stopped by the user — keep the partial reply as context.
        history.current.push({ role: 'assistant', content: reply });
        setStatus({ kind: 'idle' });
      } else {
        history.current.pop();
        setStatus({ kind: 'error', detail: e instanceof Error ? e.message : String(e) });
      }
    } finally {
      if (abort.current === controller) abort.current = null;
    }
  }, [client, draft, mode, model, native, status.kind]);

  // Test/demo hook (the RN twin of the Swift app's `-autosend`): launching
  // via `sapientchat://chat?autosend=<prompt>` sends one message on start —
  // `xcrun simctl openurl <sim> "sapientchat://chat?autosend=Hi"` drives a
  // real end-to-end on-device turn with no UI scripting.
  const sendRef = useRef(send);
  sendRef.current = send;
  useEffect(() => {
    Linking.getInitialURL().then((url) => {
      const m = url && /[?&]autosend=([^&]+)/.exec(url);
      if (m) sendRef.current(decodeURIComponent(m[1]));
    });
    const sub = Linking.addEventListener('url', ({ url }) => {
      const m = /[?&]autosend=([^&]+)/.exec(url);
      if (m) sendRef.current(decodeURIComponent(m[1]));
    });
    return () => sub.remove();
  }, []);

  const stop = useCallback(() => abort.current?.abort(), []);
  const clear = useCallback(() => {
    // Abort any in-flight stream and invalidate its turn — Clear stays
    // enabled while generating, and an orphaned stream would otherwise
    // keep appending to the fresh history.
    turn.current++;
    abort.current?.abort();
    history.current = [];
    setBubbles([]);
    setStatus({ kind: 'idle' });
  }, []);

  const subtitle =
    status.kind === 'generating'
      ? 'generating…'
      : status.kind === 'error'
        ? `error: ${status.detail}`
        : mode === 'device'
          ? `on-device${backend ? ` · ${backend}` : ''} · ${model}`
          : `via sapient serve · ${model}`;

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="auto" />
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Text style={styles.title}>SAPIENT Chat</Text>
          <Button
            title={mode === 'device' ? 'on-device' : 'server'}
            onPress={() => setMode((m) => (m === 'device' ? 'server' : 'device'))}
            disabled={status.kind === 'generating'}
          />
        </View>
        <Text style={styles.subtitle}>{subtitle}</Text>
        <View style={styles.settingsRow}>
          {mode === 'server' && (
            <TextInput
              style={[styles.settingsInput, { flex: 3 }]}
              value={baseUrl}
              onChangeText={setBaseUrl}
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="http://<dev-machine-ip>:11435"
            />
          )}
          <TextInput
            style={[styles.settingsInput, { flex: 2 }]}
            value={model}
            onChangeText={setModel}
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="model alias"
          />
        </View>
      </View>

      <FlatList
        ref={listRef}
        style={styles.transcript}
        data={bubbles}
        keyExtractor={(b) => b.id}
        onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        renderItem={({ item }) => (
          <View
            style={[
              styles.bubble,
              item.role === 'user' ? styles.userBubble : styles.assistantBubble,
            ]}
          >
            <Text>{item.text || '…'}</Text>
          </View>
        )}
      />

      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <View style={styles.inputBar}>
          <TextInput
            style={styles.input}
            value={draft}
            onChangeText={setDraft}
            placeholder="Message…"
            editable={status.kind !== 'generating'}
            onSubmitEditing={() => send()}
          />
          {status.kind === 'generating' ? (
            <Button title="Stop" onPress={stop} />
          ) : (
            <Button title="Send" onPress={() => send()} disabled={!draft.trim()} />
          )}
          <Button title="Clear" onPress={clear} disabled={bubbles.length === 0} />
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#fff' },
  header: { padding: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#ccc' },
  titleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  title: { fontSize: 17, fontWeight: '600' },
  subtitle: { fontSize: 12, color: '#666', marginTop: 2 },
  settingsRow: { flexDirection: 'row', gap: 8, marginTop: 8 },
  settingsInput: {
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#bbb',
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 6,
    fontSize: 13,
  },
  transcript: { flex: 1, paddingHorizontal: 10 },
  bubble: {
    marginVertical: 4,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    maxWidth: '85%',
  },
  userBubble: { alignSelf: 'flex-end', backgroundColor: '#d7e8ff' },
  assistantBubble: { alignSelf: 'flex-start', backgroundColor: '#f0f0f0' },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    padding: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderColor: '#ccc',
  },
  input: {
    flex: 1,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#bbb',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
});
