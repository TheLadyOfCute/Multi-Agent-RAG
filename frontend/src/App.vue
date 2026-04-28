<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  Activity,
  BarChart3,
  Bot,
  Database,
  Download,
  FileText,
  Gauge,
  MessageSquare,
  RefreshCw,
  Send,
  Sparkles,
  Trash2,
  Upload,
  X
} from 'lucide-vue-next'
import { api, type AppState, type ChatMessage, type DocumentRecord, type EvaluationQuestion, type EvaluationResult, type EvaluationScore, type TaskStatus } from './api/client'

const tabs = [
  { key: 'chat', label: '对话', icon: MessageSquare },
  { key: 'evaluation', label: '评估', icon: BarChart3 },
  { key: 'stats', label: '统计', icon: Database },
  { key: 'performance', label: '性能', icon: Gauge }
] as const

const activeTab = ref<(typeof tabs)[number]['key']>('chat')
const appState = ref<AppState | null>(null)
const documents = ref<DocumentRecord[]>([])
const messages = ref<ChatMessage[]>([])
const questions = ref<EvaluationQuestion[]>([])
const query = ref('')
const busy = ref(false)
const graphRefreshing = ref(false)
const clearingData = ref(false)
const notice = ref('')
const error = ref('')
const uploadTask = ref<TaskStatus | null>(null)
const evalTask = ref<TaskStatus | null>(null)
const chatTask = ref<TaskStatus | null>(null)
const evaluationResult = ref<EvaluationResult | null>(null)
const testFile = ref('data/test_questions.json')
const reuseRagOutputs = ref(false)
const selectedDocument = ref('')
const preview = ref<{ text: string; chars: number; words: number } | null>(null)
const openCitations = ref<Record<string, boolean>>({})

const totalChunks = computed(() => documents.value.reduce((sum, doc) => sum + Number(doc.chunks || 0), 0))
const userQueries = computed(() => messages.value.filter((message) => message.role === 'user').length)
const graphCounts = computed(() => appState.value?.neo4j.counts ?? { nodes: 0, edges: 0 })
const perf = computed(() => appState.value?.performance ?? {})
const uploadProgress = computed(() => Math.round((uploadTask.value?.progress ?? 0) * 100))
const evalProgress = computed(() => Math.round((evalTask.value?.progress ?? 0) * 100))
const chatProgress = computed(() => Math.max(4, Math.round((chatTask.value?.progress ?? 0) * 100)))
const questionRows = computed(() => {
  const scoreById = new Map<string, EvaluationScore>()
  for (const score of evaluationResult.value?.scores ?? []) {
    scoreById.set(score.id, score)
  }
  return questions.value.map((question) => {
    const score = scoreById.get(question.id)
    return {
      ...question,
      retrieved_chunk_ids: score?.retrieved_chunk_ids ?? question.retrieved_chunk_ids ?? [],
      gold_chunk_ids: score?.gold_chunk_ids ?? question.gold_chunk_ids ?? []
    }
  })
})

function setError(value: unknown) {
  error.value = value instanceof Error ? value.message : String(value)
}

function citationKey(messageIndex: number, childId: string, sourceNumber: number) {
  return `${messageIndex}:${childId || sourceNumber}`
}

function toggleCitation(messageIndex: number, childId: string, sourceNumber: number) {
  const key = citationKey(messageIndex, childId, sourceNumber)
  openCitations.value = {
    ...openCitations.value,
    [key]: !openCitations.value[key]
  }
}

async function refreshAll() {
  try {
    const [stateResult, docsResult, messagesResult] = await Promise.all([
      api.state(),
      api.documents(),
      api.messages()
    ])
    appState.value = stateResult
    documents.value = docsResult.documents
    messages.value = messagesResult.messages
    if (!selectedDocument.value && documents.value.length) {
      selectedDocument.value = documents.value[0].name
    }
  } catch (err) {
    setError(err)
  }
}

async function loadQuestions(path = testFile.value) {
  try {
    const result = await api.questions(path)
    testFile.value = result.test_file
    questions.value = result.questions
  } catch (err) {
    questions.value = []
    setError(err)
  }
}

async function onUploadDocument(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  error.value = ''
  notice.value = `已提交：${file.name}`
  try {
    const result = await api.uploadDocument(file)
    uploadTask.value = { task_id: result.task_id, status: 'pending' }
    pollTask(result.task_id, 'upload')
  } catch (err) {
    setError(err)
  } finally {
    input.value = ''
  }
}

async function pollTask(taskId: string, kind: 'upload' | 'eval' | 'chat') {
  while (true) {
    const task = await api.task(taskId)
    if (kind === 'upload') uploadTask.value = task
    if (kind === 'eval') evalTask.value = task
    if (kind === 'chat') chatTask.value = task

    if (kind !== 'chat') {
      await refreshAll()
    }

    if (task.status === 'done') {
      if (kind === 'upload') notice.value = '文档处理完成'
      if (kind === 'eval') {
        notice.value = '评估完成'
        evaluationResult.value = task.result as EvaluationResult
      }
      if (kind === 'chat') {
        notice.value = '回答已生成'
        busy.value = false
        chatTask.value = null
      }
      if (kind === 'eval') activeTab.value = 'evaluation'
      await refreshAll()
      return
    }

    if (task.status === 'error' || task.status === 'not_found') {
      error.value = task.error || '任务失败'
      if (kind === 'chat') busy.value = false
      return
    }

    await new Promise((resolve) => window.setTimeout(resolve, kind === 'chat' ? 900 : 1800))
  }
}

async function askQuestion(text?: string) {
  const question = (text ?? query.value).trim()
  if (!question || busy.value) return
  error.value = ''
  notice.value = '问题已提交，正在启动多 Agent 工作流...'
  busy.value = true
  query.value = ''
  messages.value.push({ role: 'user', content: question })
  chatTask.value = {
    task_id: 'local-pending',
    status: 'pending',
    progress: 0,
    stage: 'submitting',
    last_id: '正在提交问题'
  }
  try {
    const result = await api.startChatTask(question)
    chatTask.value = {
      task_id: result.task_id,
      status: 'pending',
      progress: 0.02,
      stage: 'queued',
      last_id: '任务已进入队列'
    }
    pollTask(result.task_id, 'chat')
  } catch (err) {
    setError(err)
    busy.value = false
    chatTask.value = null
    await refreshAll()
  }
}

async function refreshGraph() {
  if (graphRefreshing.value) return
  graphRefreshing.value = true
  error.value = ''
  notice.value = '正在刷新图数据库统计...'
  try {
    const result = await api.graphStats()
    await refreshAll()
    const nodes = result?.counts?.nodes ?? appState.value?.neo4j.counts.nodes ?? 0
    const edges = result?.counts?.edges ?? appState.value?.neo4j.counts.edges ?? 0
    notice.value = `图数据库已刷新：${nodes} 个节点，${edges} 条边`
  } catch (err) {
    setError(err)
  } finally {
    graphRefreshing.value = false
  }
}

async function removeDocument(name: string) {
  try {
    const result = await api.deleteDocument(name)
    notice.value = result.vector_chunks_deleted ? '文档已删除' : '文档记录已删除，向量数据未单独清理'
    await refreshAll()
  } catch (err) {
    setError(err)
  }
}

async function clearMessages() {
  await api.clearMessages()
  messages.value = []
  await refreshAll()
}

async function clearData() {
  if (clearingData.value) return
  clearingData.value = true
  error.value = ''
  try {
    await api.clearData()
    uploadTask.value = null
    evalTask.value = null
    evaluationResult.value = null
    chatTask.value = null
    preview.value = null
    selectedDocument.value = ''
    notice.value = '所有数据已清空'
    await refreshAll()
  } catch (err) {
    setError(err)
  } finally {
    clearingData.value = false
  }
}

async function uploadQuestionFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  try {
    const result = await api.uploadQuestions(file)
    testFile.value = result.test_file
    questions.value = result.questions
    notice.value = `已加载 ${result.questions.length} 个评估问题`
  } catch (err) {
    setError(err)
  } finally {
    input.value = ''
  }
}

async function startEvaluation() {
  try {
    evaluationResult.value = null
    const result = await api.startRagas(testFile.value, reuseRagOutputs.value)
    evalTask.value = { task_id: result.task_id, status: 'pending' }
    pollTask(result.task_id, 'eval')
  } catch (err) {
    setError(err)
  }
}

async function loadPreview() {
  if (!selectedDocument.value) return
  try {
    const result = await api.previewDocument(selectedDocument.value)
    preview.value = result
  } catch (err) {
    setError(err)
  }
}

async function savePerformance() {
  try {
    const result = await api.savePerformance()
    notice.value = `指标已保存至 ${result.path}`
    await refreshAll()
  } catch (err) {
    setError(err)
  }
}

onMounted(async () => {
  await refreshAll()
  await loadQuestions()
  window.setInterval(refreshAll, 10000)
})
</script>

<template>
  <main class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <Sparkles :size="22" />
        <div>
          <strong>智能文档问答系统</strong>
          <span>Multi-Agent RAG</span>
        </div>
      </div>

      <label class="upload-zone">
        <Upload :size="18" />
        <span>上传文档</span>
        <input type="file" accept=".pdf,.docx,.txt" @change="onUploadDocument" />
      </label>

      <div v-if="uploadTask && uploadTask.status !== 'done'" class="task-strip">
        <span>{{ uploadTask.stage || 'processing' }}</span>
        <div><i :style="{ width: `${uploadProgress}%` }" /></div>
      </div>

      <section class="sidebar-section">
        <div class="section-title">
          <FileText :size="16" />
          <span>文档</span>
          <small>{{ documents.length }}</small>
        </div>
        <div class="doc-list">
          <article v-for="doc in documents" :key="doc.name" class="doc-row">
            <button class="doc-main" @click="selectedDocument = doc.name; activeTab = 'stats'">
              <strong>{{ doc.name }}</strong>
              <span>{{ doc.chunks }} chunks · {{ doc.type }}</span>
            </button>
            <button class="icon-button danger" title="删除文档" @click="removeDocument(doc.name)">
              <Trash2 :size="15" />
            </button>
          </article>
          <p v-if="!documents.length" class="muted">暂无文档</p>
        </div>
      </section>

      <section class="sidebar-section graph-card">
        <div class="section-title">
          <Database :size="16" />
          <span>图数据库</span>
          <button
            class="icon-button"
            :class="{ spinning: graphRefreshing }"
            title="刷新图统计"
            :disabled="graphRefreshing"
            @click="refreshGraph"
          >
            <RefreshCw :size="15" />
          </button>
        </div>
        <div class="metric-grid compact">
          <div><strong>{{ graphCounts.nodes }}</strong><span>节点</span></div>
          <div><strong>{{ graphCounts.edges }}</strong><span>边</span></div>
        </div>
        <p v-if="appState?.neo4j.error" class="muted">{{ appState.neo4j.error }}</p>
        <div class="entity-list">
          <span v-for="entity in appState?.neo4j.top_entities ?? []" :key="String(entity[0])">
            {{ entity[0] }} · {{ entity[1] }}
          </span>
        </div>
      </section>

      <section class="sidebar-section">
        <div class="section-title">
          <Sparkles :size="16" />
          <span>示例问题</span>
        </div>
        <button class="sample" @click="askQuestion('这篇文档的核心观点是什么？')">这篇文档的核心观点是什么？</button>
        <button class="sample" @click="askQuestion('请总结文档中的关键技术点。')">请总结文档中的关键技术点。</button>
        <button class="sample" @click="askQuestion('文档中有哪些值得注意的风险？')">文档中有哪些值得注意的风险？</button>
      </section>

      <div class="sidebar-actions">
        <a class="secondary-button" :href="api.exportChatUrl()" download="chat_history.txt">
          <Download :size="15" /> 导出对话
        </a>
        <button class="secondary-button" @click="clearMessages">
          <X :size="15" /> 清空对话
        </button>
        <button class="secondary-button danger-text" :disabled="clearingData" @click="clearData">
          <Trash2 :size="15" /> {{ clearingData ? '清空中' : '清空数据' }}
        </button>
      </div>
    </aside>

    <section class="workspace">
      <header class="topbar">
        <div>
          <h1>智能文档问答系统</h1>
          <p>{{ appState?.restore_status || 'Ready' }}</p>
        </div>
        <div class="metric-row">
          <div><strong>{{ documents.length }}</strong><span>文档</span></div>
          <div><strong>{{ messages.length }}</strong><span>消息</span></div>
          <div><strong>{{ totalChunks }}</strong><span>块</span></div>
        </div>
      </header>

      <nav class="tabs">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          :class="{ active: activeTab === tab.key }"
          @click="activeTab = tab.key"
        >
          <component :is="tab.icon" :size="17" />
          {{ tab.label }}
        </button>
      </nav>

      <div v-if="notice" class="notice">{{ notice }}</div>
      <div v-if="error" class="notice error">{{ error }}</div>

      <section v-if="activeTab === 'chat'" class="panel chat-panel">
        <div class="messages">
          <div v-if="!messages.length && !chatTask" class="empty-chat">
            <Bot :size="30" />
            <strong>上传文档后开始提问</strong>
          </div>
          <article v-for="(message, index) in messages" :key="index" :class="['message', message.role]">
            <div class="message-head">
              <span>{{ message.role === 'user' ? 'You' : 'Assistant' }}</span>
            </div>
            <p>{{ message.content }}</p>
            <div v-if="message.citations?.length" class="citations">
              <button
                v-for="citation in message.citations"
                :key="citation.child_id"
                class="chip source-chip"
                :class="{ expanded: openCitations[citationKey(index, citation.child_id, citation.source_number)] }"
                :aria-expanded="openCitations[citationKey(index, citation.child_id, citation.source_number)] ? 'true' : 'false'"
                @click="toggleCitation(index, citation.child_id, citation.source_number)"
              >
                来源 {{ citation.source_number }} · {{ citation.filename }} · {{ citation.score.toFixed(3) }}
              </button>
            </div>
            <div v-if="message.citations?.length" class="source-panels">
              <article
                v-for="citation in message.citations"
                v-show="openCitations[citationKey(index, citation.child_id, citation.source_number)]"
                :key="`${citation.child_id}-panel`"
                class="source-panel"
              >
                <header>
                  <strong>[{{ citation.source_number }}] {{ citation.filename }}</strong>
                  <span>{{ citation.chunk_type }} · {{ citation.score.toFixed(4) }}</span>
                </header>
                <div class="source-id">{{ citation.child_id }}</div>
                <p>{{ citation.text_preview }}</p>
              </article>
            </div>
            <div v-if="message.workflow_metadata" class="metadata-grid">
              <span>strategy: {{ message.workflow_metadata.strategy }}</span>
              <span>critic: {{ message.workflow_metadata.critic_score }}</span>
              <span>rounds: {{ message.workflow_metadata.retrieval_rounds }}</span>
            </div>
          </article>
          <article v-if="chatTask" class="message assistant loading-message">
            <div class="message-head">
              <span>Assistant</span>
              <span class="loading-dots"><i /><i /><i /></span>
            </div>
            <p>{{ chatTask.last_id || '正在处理问题...' }}</p>
            <div class="task-strip wide">
              <span>{{ chatTask.stage || chatTask.status }} · {{ chatProgress }}%</span>
              <div><i :style="{ width: `${chatProgress}%` }" /></div>
            </div>
          </article>
        </div>
        <form class="command-bar" @submit.prevent="askQuestion()">
          <Sparkles :size="20" />
          <input v-model="query" :disabled="busy" placeholder="请输入您关于文档的问题..." />
          <button :disabled="busy || !query.trim()" title="发送">
            <Send :size="18" />
          </button>
        </form>
      </section>

      <section v-else-if="activeTab === 'evaluation'" class="panel stack">
        <div class="section-heading">
          <h2>RAGAS 评估</h2>
        </div>
        <p class="hint">
          用 RAGAS 对“检索 + 回答”质量打分（0～1，越高越好）：Faithfulness（回答是否忠于上下文）、Relevancy（是否贴合问题）、
          Context Precision（检索是否精确）、Context Recall（检索是否覆盖关键信息）。
        </p>
        <div class="toolbar">
          <label class="secondary-button">
            <Upload :size="15" /> 上传问题
            <input type="file" accept=".json" @change="uploadQuestionFile" />
          </label>
          <input v-model="testFile" class="path-input" />
          <button class="secondary-button" @click="loadQuestions()">刷新</button>
          <label class="check">
            <input v-model="reuseRagOutputs" type="checkbox" />
            复用 RAG 输出
          </label>
          <button class="primary-button" :disabled="!!evalTask && evalTask.status === 'running'" @click="startEvaluation">
            <Activity :size="16" /> 开始评估
          </button>
        </div>
        <div v-if="evalTask" class="task-strip wide">
          <span>{{ evalTask.status }} · {{ evalTask.stage || 'waiting' }} · {{ evalTask.current || 0 }}/{{ evalTask.total || 0 }}</span>
          <div><i :style="{ width: `${evalProgress}%` }" /></div>
        </div>
        <div v-if="evaluationResult" class="metric-grid">
          <div><strong>{{ evaluationResult.summary?.num_questions ?? '-' }}</strong><span>问题数</span></div>
          <div><strong>{{ evaluationResult.summary?.num_success ?? '-' }}</strong><span>成功</span></div>
          <div><strong>{{ evaluationResult.summary?.num_failed ?? '-' }}</strong><span>失败</span></div>
          <div><strong>{{ typeof evaluationResult.summary?.mean_overall === 'number' ? evaluationResult.summary.mean_overall.toFixed(2) : '-' }}</strong><span>Overall 平均</span></div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Question</th>
                <th>Type</th>
                <th>Reference</th>
                <th>实际命中 chunk_id</th>
                <th>参考 chunk_id</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in questionRows" :key="item.id">
                <td>{{ item.id }}</td>
                <td>{{ item.question }}</td>
                <td>{{ item.question_type }}</td>
                <td>{{ item.reference }}</td>
                <td class="id-cell">{{ item.retrieved_chunk_ids?.length ? item.retrieved_chunk_ids.join(', ') : '-' }}</td>
                <td class="id-cell">{{ item.gold_chunk_ids?.length ? item.gold_chunk_ids.join(', ') : '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <section v-if="evaluationResult" class="evaluation-results">
          <div class="section-heading">
            <h2>评估结果</h2>
          </div>
          <div class="metric-grid">
            <div><strong>{{ typeof evaluationResult.summary.mean_faithfulness === 'number' ? evaluationResult.summary.mean_faithfulness.toFixed(2) : '-' }}</strong><span>Faithfulness 平均</span></div>
            <div><strong>{{ typeof evaluationResult.summary.mean_answer_relevancy === 'number' ? evaluationResult.summary.mean_answer_relevancy.toFixed(2) : '-' }}</strong><span>Relevancy 平均</span></div>
            <div><strong>{{ typeof evaluationResult.summary.mean_context_precision === 'number' ? evaluationResult.summary.mean_context_precision.toFixed(2) : '-' }}</strong><span>Precision 平均</span></div>
            <div><strong>{{ typeof evaluationResult.summary.mean_context_recall === 'number' ? evaluationResult.summary.mean_context_recall.toFixed(2) : '-' }}</strong><span>Recall 平均</span></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Faithfulness</th>
                  <th>Relevancy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Overall</th>
                  <th>Error</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in evaluationResult.scores" :key="`${row.id}-score`">
                  <td>{{ row.id }}</td>
                  <td>{{ typeof row.faithfulness === 'number' ? row.faithfulness.toFixed(2) : '-' }}</td>
                  <td>{{ typeof row.answer_relevancy === 'number' ? row.answer_relevancy.toFixed(2) : '-' }}</td>
                  <td>{{ typeof row.context_precision === 'number' ? row.context_precision.toFixed(2) : '-' }}</td>
                  <td>{{ typeof row.context_recall === 'number' ? row.context_recall.toFixed(2) : '-' }}</td>
                  <td>{{ typeof row.overall === 'number' ? row.overall.toFixed(2) : '-' }}</td>
                  <td>{{ row.error || '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="result-paths">
            <span>RAG 输出：{{ evaluationResult.rag_outputs_path }}</span>
            <span>评分文件：{{ evaluationResult.scores_path }}</span>
            <span>汇总文件：{{ evaluationResult.summary_path }}</span>
          </div>
        </section>
      </section>

      <section v-else-if="activeTab === 'stats'" class="panel stack">
        <div class="metric-grid">
          <div><strong>{{ totalChunks }}</strong><span>总块数</span></div>
          <div><strong>{{ userQueries }}</strong><span>查询次数</span></div>
          <div><strong>{{ documents.length }}</strong><span>文档数量</span></div>
          <div><strong>{{ appState?.rag_initialized ? 'Ready' : 'Idle' }}</strong><span>系统状态</span></div>
        </div>
        <div class="toolbar">
          <select v-model="selectedDocument" class="path-input">
            <option v-for="doc in documents" :key="doc.name" :value="doc.name">{{ doc.name }}</option>
          </select>
          <button class="secondary-button" @click="loadPreview">加载预览</button>
        </div>
        <article v-if="selectedDocument" class="doc-detail">
          <h2>{{ selectedDocument }}</h2>
          <p v-if="preview">{{ preview.chars.toLocaleString() }} 字符 · {{ preview.words.toLocaleString() }} 词</p>
          <pre v-if="preview">{{ preview.text }}</pre>
        </article>
      </section>

      <section v-else class="panel stack">
        <div class="metric-grid">
          <div><strong>{{ perf.total_queries ?? 0 }}</strong><span>总查询</span></div>
          <div><strong>{{ perf.avg_latency_ms ? (perf.avg_latency_ms / 1000).toFixed(2) + 's' : '-' }}</strong><span>平均延迟</span></div>
          <div><strong>{{ perf.cache_hit_rate ? (perf.cache_hit_rate * 100).toFixed(1) + '%' : '0%' }}</strong><span>缓存命中</span></div>
          <div><strong>{{ perf.avg_chunks?.toFixed?.(1) ?? '-' }}</strong><span>平均块数</span></div>
        </div>
        <button class="primary-button fit" @click="savePerformance">
          <Download :size="16" /> 保存指标
        </button>
      </section>
    </section>
  </main>
</template>
