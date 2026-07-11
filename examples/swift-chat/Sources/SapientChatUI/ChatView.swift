import SwiftUI

/// Minimal chat UI over `ChatViewModel` — shared by the macOS executable
/// and the iOS app (see ../../iOS and project.yml).
public struct ChatView: View {
    @StateObject private var model = ChatViewModel()
    @State private var draft = ""

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            inputBar
        }
        #if os(macOS)
        .frame(minWidth: 480, minHeight: 560)
        #endif
        .onAppear(perform: autosendIfRequested)
    }

    /// Test/demo hook: `-autosend "<prompt>"` in the launch arguments sends
    /// one message on appear — lets `simctl launch` drive a real end-to-end
    /// turn on a simulator with no UI scripting.
    private func autosendIfRequested() {
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "-autosend"), args.indices.contains(idx + 1) {
            model.send(args[idx + 1])
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("SAPIENT Chat").font(.headline)
                Text(statusLine).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            TextField("model alias", text: $model.modelAlias)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 180)
                .disabled(model.isBusy)
            Button("Clear") { model.clearConversation() }
                .disabled(model.isBusy || model.messages.isEmpty)
        }
        .padding(10)
    }

    private var statusLine: String {
        switch model.status {
        case .idle:
            let backend = model.backendLabel.map { " · \($0)" } ?? ""
            return "on-device\(backend)"
        case .loading(let alias):
            return "loading \(alias) — first run downloads the model…"
        case .generating:
            return "generating…"
        case .failed(let message):
            return "error: \(message)"
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(model.messages) { message in
                        bubble(message)
                            .frame(
                                maxWidth: .infinity,
                                alignment: message.role == .user ? .trailing : .leading
                            )
                    }
                }
                .padding(10)
            }
            .onChange(of: model.messages.last?.text) { _ in
                if let last = model.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private func bubble(_ message: DisplayMessage) -> some View {
        Text(message.text.isEmpty ? "…" : message.text)
            .textSelection(.enabled)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                message.role == .user
                    ? AnyShapeStyle(Color.accentColor.opacity(0.18))
                    : AnyShapeStyle(.quaternary),
                in: RoundedRectangle(cornerRadius: 12)
            )
            .id(message.id)
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message…", text: $draft)
                .textFieldStyle(.roundedBorder)
                .onSubmit(submit)
                .disabled(model.isBusy)
            if case .generating = model.status {
                Button("Stop") { model.stop() }
            } else {
                Button("Send", action: submit)
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isBusy
                        || draft.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(10)
    }

    private func submit() {
        model.send(draft)
        draft = ""
    }
}
