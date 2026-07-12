package so.openhorizon.sapient.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import uniffi.sapient_ffi.GenerationOptions
import uniffi.sapient_ffi.LlmSession
import uniffi.sapient_ffi.TokenListener

data class DisplayMessage(val role: Role, val text: String) {
    enum class Role { USER, ASSISTANT }
}

sealed interface Status {
    data object Idle : Status
    /** First send downloads + loads the model — can take minutes cold. */
    data class Loading(val model: String) : Status
    data object Generating : Status
    data class Failed(val message: String) : Status
}

class ChatViewModel : ViewModel() {
    val messages = MutableStateFlow<List<DisplayMessage>>(emptyList())
    val status = MutableStateFlow<Status>(Status.Idle)
    val backendLabel = MutableStateFlow<String?>(null)

    /** Dev default per docs/MOBILE.md §5.2 — smallest model until boring. */
    val modelAlias = MutableStateFlow("smollm2-135m-q4")

    private var session: LlmSession? = null
    private var loadedAlias: String? = null
    // Per-send flag: a new send must NOT reopen an old stream's cancel gate,
    // so each turn gets its own; this points at the active turn's.
    private var activeCancel: AtomicBoolean? = null
    // Bumped by every send and by Clear — stale writes from an old stream
    // (tokens, status flips) compare against it and drop.
    private val turnEpoch = AtomicInteger(0)

    val isBusy: StateFlow<Status> get() = status

    fun send(text: String) {
        val prompt = text.trim()
        if (prompt.isEmpty() || status.value is Status.Loading || status.value is Status.Generating) return

        messages.update { it + DisplayMessage(DisplayMessage.Role.USER, prompt) +
            DisplayMessage(DisplayMessage.Role.ASSISTANT, "") }
        val replyIndex = messages.value.lastIndex

        val alias = modelAlias.value
        val needsLoad = session == null || loadedAlias != alias
        status.value = if (needsLoad) Status.Loading(alias) else Status.Generating

        val epoch = turnEpoch.incrementAndGet()
        val cancelled = AtomicBoolean(false)
        activeCancel = cancelled

        val listener = object : TokenListener {
            override fun onToken(token: String): Boolean {
                messages.update { msgs ->
                    // The transcript can be cleared (and refilled by a new
                    // turn) while a stream drains — drop stale tokens.
                    if (turnEpoch.get() != epoch || replyIndex > msgs.lastIndex) return@update msgs
                    msgs.toMutableList().also {
                        it[replyIndex] = it[replyIndex].copy(text = it[replyIndex].text + token)
                    }
                }
                return !cancelled.get() // false = cancel generation engine-side
            }
        }

        // The FFI is blocking by design — keep it on the IO dispatcher.
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val active = if (needsLoad) {
                    // Greedy decoding: deterministic, least drift for tiny dev models
                    // (docs/MOBILE.md §5.2); add temperature only with 0.5B+ models.
                    LlmSession.load(alias, GenerationOptions(maxTokens = 512u))
                        .also {
                            session = it
                            loadedAlias = alias
                            backendLabel.value = it.backendLabel()
                        }
                } else {
                    session!!
                }
                if (turnEpoch.get() == epoch) status.value = Status.Generating
                active.chatStream(prompt, listener)
                if (turnEpoch.get() == epoch) status.value = Status.Idle
            } catch (e: Exception) {
                if (turnEpoch.get() != epoch) return@launch
                messages.update { msgs ->
                    if (replyIndex > msgs.lastIndex) return@update msgs
                    msgs.toMutableList().also { it[replyIndex] = it[replyIndex].copy(text = "") }
                }
                status.value = Status.Failed(e.message ?: e.toString())
            }
        }
    }

    /** Stop mid-reply; the partial text stays in the transcript by design. */
    fun stop() {
        activeCancel?.set(true)
    }

    fun clearConversation() {
        // Stop any in-flight stream and invalidate its epoch so a late token
        // or status flip from the old turn can't touch the fresh transcript.
        turnEpoch.incrementAndGet()
        activeCancel?.set(true)
        session?.reset()
        messages.value = emptyList()
        status.value = Status.Idle
    }
}
