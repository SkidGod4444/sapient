package so.openhorizon.sapient.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import java.util.concurrent.atomic.AtomicBoolean
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
    private val cancelled = AtomicBoolean(false)

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
        cancelled.set(false)

        val listener = object : TokenListener {
            override fun onToken(token: String): Boolean {
                messages.update { msgs ->
                    // The transcript can be cleared mid-stream — never index blindly.
                    if (replyIndex > msgs.lastIndex) return@update msgs
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
                    LlmSession.load(alias, GenerationOptions(maxTokens = 512u, temperature = 0.7f))
                        .also {
                            session = it
                            loadedAlias = alias
                            backendLabel.value = it.backendLabel()
                        }
                } else {
                    session!!
                }
                status.value = Status.Generating
                active.chatStream(prompt, listener)
                status.value = Status.Idle
            } catch (e: Exception) {
                messages.update { msgs ->
                    if (replyIndex > msgs.lastIndex) return@update msgs
                    msgs.toMutableList().also { it[replyIndex] = it[replyIndex].copy(text = "") }
                }
                status.value = Status.Failed(e.message ?: e.toString())
            }
        }
    }

    /** Stop mid-reply; the partial text stays in the transcript by design. */
    fun stop() = cancelled.set(true)

    fun clearConversation() {
        // Stop any in-flight stream first so a late token can't target a
        // reply bubble that no longer exists (the UI disables Clear while
        // busy, but the stream outlives the tap by at least one token).
        cancelled.set(true)
        session?.reset()
        messages.value = emptyList()
        status.value = Status.Idle
    }
}
