import Foundation
import Sapient

/// One rendered message bubble.
public struct DisplayMessage: Identifiable, Equatable {
    public enum Role { case user, assistant }
    public let id = UUID()
    public let role: Role
    public var text: String
}

/// Thread-safe cancellation flag shared with the streaming listener.
final class Cancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func cancel() { lock.lock(); flag = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return flag }
}

/// Bridges the blocking FFI token callback onto the main actor.
/// Returning `false` from `onToken` cancels generation engine-side.
final class StreamListener: TokenListener, @unchecked Sendable {
    private let cancellation: Cancellation
    private let deliver: @Sendable (String) -> Void
    init(cancellation: Cancellation, deliver: @escaping @Sendable (String) -> Void) {
        self.cancellation = cancellation
        self.deliver = deliver
    }
    func onToken(token: String) -> Bool {
        deliver(token)
        return !cancellation.isCancelled
    }
}

@MainActor
public final class ChatViewModel: ObservableObject {
    public enum Status: Equatable {
        case idle
        /// First send downloads + loads the model — can take minutes cold.
        case loading(model: String)
        case generating
        case failed(String)
    }

    @Published public private(set) var messages: [DisplayMessage] = []
    @Published public private(set) var status: Status = .idle
    @Published public private(set) var backendLabel: String?
    /// Dev default per docs/MOBILE.md §5.2: iterate on the smallest model;
    /// switch to e.g. `llama3.2-1b-q4` only once the plumbing is boring.
    @Published public var modelAlias = "smollm2-135m-q4"

    private var session: LlmSession?
    private var loadedAlias: String?
    private var cancellation: Cancellation?
    /// The FFI is blocking by design — keep it off the main thread and off
    /// the Swift cooperative pool. Serial: one turn at a time.
    private let inferenceQueue = DispatchQueue(label: "so.openhorizon.sapient.inference")

    public init() {
        // Keep model downloads inside the app sandbox so the OS can reclaim
        // them and uninstall removes them (docs/MOBILE.md §5.4).
        if let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            setenv("HF_HOME", caches.appendingPathComponent("sapient").path, 1)
        }
    }

    public var isBusy: Bool {
        if case .idle = status { return false }
        if case .failed = status { return false }
        return true
    }

    public func send(_ text: String) {
        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isBusy else { return }

        messages.append(DisplayMessage(role: .user, text: prompt))
        messages.append(DisplayMessage(role: .assistant, text: ""))
        let replyIndex = messages.count - 1

        let alias = modelAlias
        let needsLoad = session == nil || loadedAlias != alias
        status = needsLoad ? .loading(model: alias) : .generating

        let cancellation = Cancellation()
        self.cancellation = cancellation
        let listener = StreamListener(cancellation: cancellation) { [weak self] token in
            DispatchQueue.main.async { self?.messages[replyIndex].text += token }
        }

        // Capture the resolved session ON the main actor — the queue closure
        // must not read isolated state (older Swift compilers reject it, and
        // Swift 6 makes it a hard error).
        let existingSession: LlmSession? = needsLoad ? nil : session

        inferenceQueue.async { [weak self] in
            do {
                let active: LlmSession
                if let existing = existingSession {
                    active = existing
                } else {
                    active = try LlmSession.load(
                        model: alias,
                        options: GenerationOptions(maxTokens: 512, temperature: 0.7)
                    )
                    DispatchQueue.main.async {
                        self?.session = active
                        self?.loadedAlias = alias
                        self?.backendLabel = active.backendLabel()
                    }
                }
                DispatchQueue.main.async { self?.status = .generating }
                _ = try active.chatStream(userMessage: prompt, listener: listener)
                DispatchQueue.main.async { self?.status = .idle }
            } catch {
                DispatchQueue.main.async {
                    self?.messages[replyIndex].text = ""
                    self?.status = .failed("\(error)")
                }
            }
        }
    }

    /// Ask the engine to stop mid-reply; the partial text stays in the
    /// transcript (and in the session history — intentional, see sapient-ffi).
    public func stop() {
        cancellation?.cancel()
    }

    public func clearConversation() {
        session?.reset()
        messages.removeAll()
        status = .idle
    }
}
