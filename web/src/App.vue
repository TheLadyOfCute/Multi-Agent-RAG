<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  Activity,
  BarChart3,
  Bot,
  BrainCircuit,
  Database,
  Download,
  FileJson,
  FileText,
  Loader2,
  MessageSquare,
  RefreshCcw,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Trash2,
  UploadCloud,
  Workflow,
  X
} from "lucide-vue-next";
import type {
  ChatMessage,
  DocumentRecord,
  PerformanceStats,
  PreviewDocument,
  RagasResult,
  SystemState,
  TaskState,
  TestQuestion
} from "./api/client";
import { api } from "./api/client";

type TaskKind = "upload" | "chat" | "ragas";

const health = ref<"checking" | "ok" | "down">("checking");
const state = ref<SystemState | null>(null);
const documents = ref<DocumentRecord[]>([]);
const messages = ref<ChatMessage[]>([]);
const preview = ref<PreviewDocument | null>(null);
const previewOpen = ref(false);
const questions = ref<TestQuestion[]>([]);
const testFile = ref("data/test_questions.json");
const ragasResult = ref<RagasResult | null>(null);
const query = ref("");
const notice = ref("");
const errorMessage = ref("");
const reuseRagOutputs = ref(false);
const confirmClearData = ref(false);
const expandedCitations = ref<Record<number, boolean>>({});
const deletingName = ref("");

const uploadTask = ref<TaskState | null>(null);
const chatTask = ref<TaskState | null>(null);
const ragasTask = ref<TaskState<RagasResult> | null>(null);
const pollTimers = new Map<TaskKind, number>();
let refreshTimer: number | undefined;

const canAsk = computed(() => Boolean(state.value?.rag_initialized) && !isBusy(chatTask.value));
const neo4jCounts = computed(() => state.value?.neo4j?.counts || {});
const performance = computed<PerformanceStats>(() => state.value?.performance || {});
const assistantCount = computed(() => messages.value.filter((item) => item.role === "assistant").length);
const latestAssistant = computed(() => [...messages.value].reverse().find((item) => item.role === "assistant"));
const summaryEntries = computed(() => Object.entries(ragasResult.value?.summary || {}).slice(0, 10));
const scoreRows = computed(() => ragasResult.value?.scores?.slice(0, 8) || []);
const scoreColumns = computed(() => {
  const row = scoreRows.value[0];
  return row ? Object.keys(row).slice(0, 7) : [];
});

onMounted(async () => {
  await refreshAll();
  refreshTimer = window.setInterval(() => {
    void refreshState();
  }, 8000);
});

onUnmounted(() => {
  if (refreshTimer) {
    window.clearInterval(refreshTimer);
  }
  for (const timer of pollTimers.values()) {
    window.clearTimeout(timer);
  }
});

async function refreshAll() {
  await Promise.allSettled([checkHealth(), refreshState(), refreshDocuments(), refreshMessages()]);
}

async function checkHealth() {
  try {
    await api.health();
    health.value = "ok";
  } catch {
    health.value = "down";
  }
}

async function refreshState() {
  try {
    state.value = await api.state();
    errorMessage.value = "";
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function refreshDocuments() {
  try {
    documents.value = (await api.documents()).documents;
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function refreshMessages() {
  try {
    messages.value = (await api.messages()).messages;
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function refreshGraph() {
  try {
    await api.graphStats();
    await refreshState();
    notice.value = "Neo4j 图统计已刷新";
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function refreshPerformance() {
  try {
    const nextPerformance = await api.performance();
    state.value = state.value ? { ...state.value, performance: nextPerformance } : state.value;
    notice.value = "性能指标已刷新";
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function savePerformance() {
  try {
    await api.savePerformance();
    notice.value = "性能指标已保存到 data/metrics.json";
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function onUploadDocument(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;

  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !["pdf", "docx", "txt"].includes(extension)) {
    errorMessage.value = "仅支持上传 pdf、docx、txt 文件";
    return;
  }

  try {
    const response = await api.uploadDocument(file);
    notice.value = `已提交文档任务：${file.name}`;
    pollTask("upload", response.task_id, uploadTask, async () => {
      await Promise.all([refreshState(), refreshDocuments()]);
    });
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function deleteDocument(name: string) {
  if (deletingName.value !== name) {
    deletingName.value = name;
    return;
  }

  try {
    await api.deleteDocument(name);
    deletingName.value = "";
    preview.value = null;
    await Promise.all([refreshState(), refreshDocuments()]);
    notice.value = `已删除 ${name}`;
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function openPreview(name: string) {
  try {
    preview.value = await api.previewDocument(name);
    previewOpen.value = true;
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function clearAllData() {
  if (!confirmClearData.value) {
    confirmClearData.value = true;
    return;
  }

  try {
    await api.clearData();
    confirmClearData.value = false;
    preview.value = null;
    ragasResult.value = null;
    questions.value = [];
    await refreshAll();
    notice.value = "文档、索引、状态和任务记录已清空";
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function sendQuestion() {
  const text = query.value.trim();
  if (!text || !canAsk.value) return;

  query.value = "";
  try {
    const response = await api.createChatTask(text);
    await refreshMessages();
    pollTask("chat", response.task_id, chatTask, async () => {
      expandedCitations.value = {};
      await Promise.all([refreshState(), refreshMessages()]);
    });
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function clearChat() {
  try {
    await api.clearMessages();
    messages.value = [];
    await refreshState();
    notice.value = "对话已清空";
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function exportChat() {
  try {
    const content = await api.exportChat();
    downloadText("multi-agent-rag-chat.txt", content);
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function loadQuestions() {
  try {
    const response = await api.evaluationQuestions(testFile.value);
    testFile.value = response.test_file;
    questions.value = response.questions;
    notice.value = `已加载 ${response.questions.length} 个测试问题`;
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function onUploadQuestions(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".json")) {
    errorMessage.value = "测试问题文件必须是 JSON";
    return;
  }

  try {
    const response = await api.uploadEvaluationQuestions(file);
    testFile.value = response.test_file;
    questions.value = response.questions;
    notice.value = `已上传 ${response.questions.length} 个测试问题`;
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

async function startRagas() {
  try {
    const response = await api.startRagas(testFile.value, reuseRagOutputs.value);
    ragasResult.value = null;
    pollTask<RagasResult>("ragas", response.task_id, ragasTask, async (task) => {
      ragasResult.value = task.result || null;
      await refreshState();
    });
  } catch (error) {
    errorMessage.value = getError(error);
  }
}

function pollTask<T>(
  kind: TaskKind,
  taskId: string,
  target: { value: TaskState<T> | null },
  onDone: (task: TaskState<T>) => Promise<void>
) {
  const existingTimer = pollTimers.get(kind);
  if (existingTimer) {
    window.clearTimeout(existingTimer);
  }

  const poll = async () => {
    try {
      const task = await api.task<T>(taskId);
      target.value = task;
      if (task.status === "done") {
        pollTimers.delete(kind);
        await onDone(task);
        return;
      }
      if (task.status === "error" || task.status === "not_found") {
        pollTimers.delete(kind);
        errorMessage.value = task.error || `任务状态异常：${task.status}`;
        return;
      }
      const timer = window.setTimeout(poll, 1000);
      pollTimers.set(kind, timer);
    } catch (error) {
      pollTimers.delete(kind);
      errorMessage.value = getError(error);
    }
  };

  void poll();
}

function toggleCitation(index: number) {
  expandedCitations.value = {
    ...expandedCitations.value,
    [index]: !expandedCitations.value[index]
  };
}

function isBusy(task: TaskState | null) {
  return task?.status === "pending" || task?.status === "running";
}

function statusText(value: boolean | undefined) {
  return value ? "Online" : "Standby";
}

function formatPercent(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "0%";
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: unknown, digits = 0) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  }).format(value);
}

function shortText(value: unknown) {
  if (value === null || value === undefined || value === "") return "--";
  if (typeof value === "number") return formatNumber(value, value % 1 === 0 ? 0 : 3);
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function getError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
</script>

<template>
  <main class="app-shell">
    <section class="topbar">
      <div>
        <p class="eyebrow"><Sparkles :size="14" /> Multi-Agent RAG</p>
        <h1>Luminous Knowledge Engine</h1>
      </div>

      <div class="status-grid">
        <div class="status-pill" :class="health">
          <Activity :size="16" />
          <span>API {{ health === "ok" ? "Online" : health === "down" ? "Offline" : "Checking" }}</span>
        </div>
        <div class="status-pill" :class="{ active: state?.rag_initialized }">
          <BrainCircuit :size="16" />
          <span>RAG {{ statusText(state?.rag_initialized) }}</span>
        </div>
        <div class="status-pill" :class="{ active: state?.task_running }">
          <Workflow :size="16" />
          <span>{{ state?.task_running ? "Task Running" : "No Task" }}</span>
        </div>
      </div>
    </section>

    <section v-if="errorMessage || notice" class="signal-stack">
      <div v-if="errorMessage" class="signal error">
        <ShieldAlert :size="18" />
        <span>{{ errorMessage }}</span>
        <button class="icon-button" title="关闭" @click="errorMessage = ''"><X :size="16" /></button>
      </div>
      <div v-if="notice" class="signal">
        <Sparkles :size="18" />
        <span>{{ notice }}</span>
        <button class="icon-button" title="关闭" @click="notice = ''"><X :size="16" /></button>
      </div>
    </section>

    <section class="overview-band">
      <div class="metric-block">
        <span>Documents</span>
        <strong>{{ state?.document_count ?? documents.length }}</strong>
      </div>
      <div class="metric-block">
        <span>Messages</span>
        <strong>{{ state?.message_count ?? messages.length }}</strong>
      </div>
      <div class="metric-block">
        <span>Answers</span>
        <strong>{{ assistantCount }}</strong>
      </div>
      <div class="metric-block wide">
        <span>Restore</span>
        <strong>{{ state?.restore_status || "Ready" }}</strong>
      </div>
    </section>

    <div class="workspace-grid">
      <aside class="panel document-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow"><Database :size="14" /> Corpus</p>
            <h2>文档库</h2>
          </div>
          <label class="primary-button upload-button" title="上传文档">
            <UploadCloud :size="17" />
            <span>Upload</span>
            <input type="file" accept=".pdf,.docx,.txt" @change="onUploadDocument" />
          </label>
        </div>

        <div v-if="uploadTask" class="task-card">
          <div class="task-card-top">
            <span>{{ uploadTask.stage || uploadTask.status }}</span>
            <strong>{{ formatPercent(uploadTask.progress) }}</strong>
          </div>
          <div class="progress-track"><span :style="{ width: formatPercent(uploadTask.progress) }" /></div>
          <p>{{ uploadTask.last_id || uploadTask.type || uploadTask.task_id }}</p>
        </div>

        <div class="document-list">
          <article v-for="doc in documents" :key="doc.name" class="document-item">
            <button class="document-main" @click="openPreview(doc.name)">
              <FileText :size="19" />
              <span>
                <strong>{{ doc.name }}</strong>
                <small>{{ doc.type || "DOC" }} · {{ doc.chunks ?? 0 }} chunks · {{ doc.pages ?? 0 }} pages</small>
              </span>
            </button>
            <button class="icon-button danger" :title="deletingName === doc.name ? '再次点击确认删除' : '删除文档'" @click="deleteDocument(doc.name)">
              <Trash2 :size="16" />
            </button>
          </article>
          <div v-if="!documents.length" class="empty-state">
            <Database :size="28" />
            <p>上传文档后，向量库、BM25 和图谱状态会在这里恢复。</p>
          </div>
        </div>

        <button class="secondary-button danger-zone" @click="clearAllData">
          <Trash2 :size="16" />
          {{ confirmClearData ? "再次点击清空全部数据" : "清空全部数据" }}
        </button>
      </aside>

      <section class="panel chat-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow"><MessageSquare :size="14" /> Conversation</p>
            <h2>智能问答</h2>
          </div>
          <div class="toolbar">
            <button class="tertiary-button" title="刷新对话" @click="refreshMessages"><RefreshCcw :size="16" /></button>
            <button class="tertiary-button" title="导出对话" @click="exportChat"><Download :size="16" /></button>
            <button class="tertiary-button" title="清空对话" @click="clearChat"><Trash2 :size="16" /></button>
          </div>
        </div>

        <div class="chat-stream">
          <article v-for="(message, index) in messages" :key="`${message.role}-${index}`" class="message" :class="message.role">
            <div class="message-avatar">
              <Bot v-if="message.role === 'assistant'" :size="17" />
              <MessageSquare v-else :size="17" />
            </div>
            <div class="message-body">
              <p>{{ message.content }}</p>

              <div v-if="message.citations?.length" class="citation-list">
                <button
                  v-for="citation in message.citations"
                  :key="`${index}-${citation.source_number}`"
                  class="citation-chip"
                  @click="toggleCitation(index * 100 + citation.source_number)"
                >
                  <Search :size="14" />
                  Source {{ citation.source_number }} · {{ citation.filename }} · {{ formatNumber(citation.score, 3) }}
                </button>
                <div
                  v-for="citation in message.citations"
                  v-show="expandedCitations[index * 100 + citation.source_number]"
                  :key="`preview-${index}-${citation.source_number}`"
                  class="citation-preview"
                >
                  <strong>{{ citation.chunk_id || citation.filename }}</strong>
                  <p>{{ citation.text_preview }}</p>
                </div>
              </div>

              <div v-if="message.workflow_metadata" class="metadata-grid">
                <span>strategy {{ shortText(message.workflow_metadata.strategy) }}</span>
                <span>rounds {{ shortText(message.workflow_metadata.retrieval_rounds) }}</span>
                <span>validation {{ shortText(message.workflow_metadata.validation_score) }}</span>
                <span>critic {{ shortText(message.workflow_metadata.critic_score) }}</span>
              </div>
            </div>
          </article>

          <div v-if="isBusy(chatTask)" class="thinking-block">
            <Loader2 class="spin" :size="18" />
            <span>{{ chatTask?.stage || "Multi-agent workflow running" }}</span>
            <strong>{{ formatPercent(chatTask?.progress) }}</strong>
          </div>

          <div v-if="!messages.length" class="empty-state chat-empty">
            <BrainCircuit :size="34" />
            <p>完成文档上传后，可以从这里发起异步多 Agent 检索问答。</p>
          </div>
        </div>

        <form class="composer" @submit.prevent="sendQuestion">
          <textarea
            v-model="query"
            :disabled="!state?.rag_initialized || isBusy(chatTask)"
            placeholder="输入一个和文档相关的问题..."
            rows="3"
          />
          <button class="primary-button send-button" :disabled="!query.trim() || !canAsk">
            <Send :size="17" />
            <span>Send</span>
          </button>
        </form>
      </section>

      <aside class="side-stack">
        <section class="panel insight-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow"><BarChart3 :size="14" /> Telemetry</p>
              <h2>系统分析</h2>
            </div>
            <button class="tertiary-button" title="刷新状态" @click="refreshAll"><RefreshCcw :size="16" /></button>
          </div>

          <div class="metric-list">
            <div><span>Neo4j</span><strong>{{ state?.neo4j.available ? "Available" : "Unavailable" }}</strong></div>
            <div><span>Nodes</span><strong>{{ neo4jCounts.nodes ?? 0 }}</strong></div>
            <div><span>Edges</span><strong>{{ neo4jCounts.edges ?? neo4jCounts.relationships ?? 0 }}</strong></div>
            <div><span>Cache Hit Rate</span><strong>{{ formatPercent(performance.cache_hit_rate) }}</strong></div>
            <div><span>Avg Latency</span><strong>{{ formatNumber(performance.avg_latency_ms, 0) }} ms</strong></div>
            <div><span>Total Queries</span><strong>{{ performance.total_queries ?? 0 }}</strong></div>
          </div>

          <div class="toolbar full">
            <button class="secondary-button" @click="refreshGraph"><Workflow :size="16" /> Graph</button>
            <button class="secondary-button" @click="refreshPerformance"><Activity :size="16" /> Perf</button>
            <button class="secondary-button" @click="savePerformance"><Download :size="16" /> Save</button>
          </div>
        </section>

        <section class="panel latest-panel">
          <p class="eyebrow"><Bot :size="14" /> Latest Answer</p>
          <p>{{ latestAssistant?.content || "还没有助手回答。" }}</p>
        </section>
      </aside>
    </div>

    <section class="panel evaluation-panel">
      <div class="panel-heading">
        <div>
          <p class="eyebrow"><FileJson :size="14" /> RAGAS</p>
          <h2>评估工作台</h2>
        </div>
        <div class="toolbar">
          <button class="secondary-button" @click="loadQuestions"><RefreshCcw :size="16" /> Load</button>
          <label class="secondary-button upload-button">
            <UploadCloud :size="16" />
            JSON
            <input type="file" accept=".json" @change="onUploadQuestions" />
          </label>
          <button class="primary-button" :disabled="isBusy(ragasTask)" @click="startRagas">
            <Activity v-if="!isBusy(ragasTask)" :size="16" />
            <Loader2 v-else class="spin" :size="16" />
            Run
          </button>
        </div>
      </div>

      <div class="evaluation-controls">
        <label>
          <span>Test file</span>
          <input v-model="testFile" type="text" />
        </label>
        <label class="toggle-row">
          <input v-model="reuseRagOutputs" type="checkbox" />
          <span>复用已有 RAG 输出</span>
        </label>
      </div>

      <div v-if="ragasTask" class="task-card evaluation-task">
        <div class="task-card-top">
          <span>{{ ragasTask.stage || ragasTask.status }}</span>
          <strong>{{ formatPercent(ragasTask.progress) }}</strong>
        </div>
        <div class="progress-track"><span :style="{ width: formatPercent(ragasTask.progress) }" /></div>
        <p>{{ ragasTask.current || 0 }} / {{ ragasTask.total || 0 }} · {{ ragasTask.last_id || ragasTask.task_id }}</p>
      </div>

      <div class="evaluation-grid">
        <div class="question-list">
          <article v-for="item in questions.slice(0, 8)" :key="item.id" class="question-item">
            <strong>{{ item.id }}</strong>
            <p>{{ item.question }}</p>
            <small>{{ item.question_type || "general" }} · {{ item.reference || "no reference" }}</small>
          </article>
          <div v-if="!questions.length" class="empty-state">
            <FileJson :size="28" />
            <p>加载或上传测试问题后，评估队列会显示在这里。</p>
          </div>
        </div>

        <div class="result-panel">
          <div v-if="summaryEntries.length" class="summary-grid">
            <div v-for="[key, value] in summaryEntries" :key="key">
              <span>{{ key }}</span>
              <strong>{{ shortText(value) }}</strong>
            </div>
          </div>

          <div v-if="scoreRows.length" class="score-table-wrap">
            <table>
              <thead>
                <tr>
                  <th v-for="column in scoreColumns" :key="column">{{ column }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, index) in scoreRows" :key="index">
                  <td v-for="column in scoreColumns" :key="column">{{ shortText(row[column]) }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div v-if="ragasResult" class="path-list">
            <span v-if="ragasResult.csv_path">CSV: {{ ragasResult.csv_path }}</span>
            <span v-if="ragasResult.jsonl_path">JSONL: {{ ragasResult.jsonl_path }}</span>
            <span v-if="ragasResult.summary_path">Summary: {{ ragasResult.summary_path }}</span>
          </div>

          <div v-if="!ragasResult" class="empty-state">
            <BarChart3 :size="28" />
            <p>RAGAS 完成后，summary、scores 和输出路径会显示在这里。</p>
          </div>
        </div>
      </div>
    </section>

    <aside v-if="previewOpen && preview" class="preview-drawer">
      <div class="panel-heading">
        <div>
          <p class="eyebrow"><FileText :size="14" /> Preview</p>
          <h2>{{ preview.name }}</h2>
        </div>
        <button class="icon-button" title="关闭预览" @click="previewOpen = false"><X :size="18" /></button>
      </div>
      <div class="metadata-grid">
        <span>{{ preview.chars }} chars</span>
        <span>{{ preview.words }} words</span>
      </div>
      <pre>{{ preview.text }}</pre>
    </aside>
  </main>
</template>
