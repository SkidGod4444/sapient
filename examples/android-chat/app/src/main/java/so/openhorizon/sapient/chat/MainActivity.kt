package so.openhorizon.sapient.chat

import android.os.Bundle
import android.system.Os
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Keep model downloads inside the app sandbox so the OS can reclaim
        // them and uninstall removes them (docs/MOBILE.md §5.4). Must happen
        // before the first LlmSession.load().
        Os.setenv("HF_HOME", File(cacheDir, "sapient").absolutePath, true)
        setContent { MaterialTheme { ChatScreen() } }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(model: ChatViewModel = viewModel()) {
    val messages by model.messages.collectAsState()
    val status by model.status.collectAsState()
    val backend by model.backendLabel.collectAsState()
    val alias by model.modelAlias.collectAsState()
    var draft by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    val busy = status is Status.Loading || status is Status.Generating

    LaunchedEffect(messages.lastOrNull()?.text) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("SAPIENT Chat")
                        Text(
                            when (val s = status) {
                                is Status.Idle -> "on-device" + (backend?.let { " · $it" } ?: "")
                                is Status.Loading -> "loading ${s.model} — first run downloads…"
                                is Status.Generating -> "generating…"
                                is Status.Failed -> "error: ${s.message}"
                            },
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                },
                actions = {
                    TextButton(
                        onClick = { model.clearConversation() },
                        enabled = !busy && messages.isNotEmpty()
                    ) { Text("Clear") }
                }
            )
        },
        bottomBar = {
            Row(
                Modifier.fillMaxWidth().padding(10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = draft,
                    onValueChange = { draft = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Message…") },
                    enabled = !busy
                )
                if (status is Status.Generating) {
                    Button(onClick = { model.stop() }) { Text("Stop") }
                } else {
                    Button(
                        onClick = { model.send(draft); draft = "" },
                        enabled = !busy && draft.isNotBlank()
                    ) { Text("Send") }
                }
            }
        }
    ) { padding ->
        LazyColumn(
            state = listState,
            modifier = Modifier.padding(padding).fillMaxSize(),
            contentPadding = PaddingValues(10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages) { message ->
                val isUser = message.role == DisplayMessage.Role.USER
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
                ) {
                    Surface(
                        color = if (isUser) MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.medium
                    ) {
                        Text(
                            message.text.ifEmpty { "…" },
                            Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
                        )
                    }
                }
            }
        }
    }
}
