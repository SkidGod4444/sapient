// SAPIENT chat over the TypeScript SDK — the React Native "rung 0" dev loop
// from docs/MOBILE.md: inference runs on `sapient serve` (your dev machine /
// server / Pi), the phone only renders. Streaming uses expo/fetch because
// RN's built-in fetch cannot stream response bodies.
//
//   sapient serve                       # on your dev machine
//   npm install && npm start           # here; point Base URL at its LAN IP
import { SapientClient, type ChatMessage } from '@openhorizon/sapient';
import { fetch as expoFetch } from 'expo/fetch';
import { StatusBar } from 'expo-status-bar';
import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
  Button,
  FlatList,
  KeyboardAvoidingView,
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

export default function App() {
  // Simulators/emulators can reach the host via localhost; a physical phone
  // needs the dev machine's LAN IP (same Wi-Fi).
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:11435');
  const [model, setModel] = useState('qwen2.5-0.5b');
  const [draft, setDraft] = useState('');
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const history = useRef<ChatMessage[]>([]);
  const abort = useRef<AbortController | null>(null);
  // Bumped by every send and by Clear — stale writes from an old stream
  // (status flips, history pushes) compare against it and drop.
  const turn = useRef(0);
  const listRef = useRef<FlatList<Bubble>>(null);

  const client = useMemo(
    () => new SapientClient({ baseUrl, fetch: expoFetch as unknown as typeof fetch }),
    [baseUrl],
  );

  const send = useCallback(async () => {
    const prompt = draft.trim();
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
  }, [client, draft, model, status.kind]);

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

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="auto" />
      <View style={styles.header}>
        <Text style={styles.title}>SAPIENT Chat</Text>
        <Text style={styles.subtitle}>
          {status.kind === 'generating'
            ? 'generating…'
            : status.kind === 'error'
              ? `error: ${status.detail}`
              : `via sapient serve · ${model}`}
        </Text>
        <View style={styles.settingsRow}>
          <TextInput
            style={[styles.settingsInput, { flex: 3 }]}
            value={baseUrl}
            onChangeText={setBaseUrl}
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="http://<dev-machine-ip>:11435"
          />
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
            onSubmitEditing={send}
          />
          {status.kind === 'generating' ? (
            <Button title="Stop" onPress={stop} />
          ) : (
            <Button title="Send" onPress={send} disabled={!draft.trim()} />
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
