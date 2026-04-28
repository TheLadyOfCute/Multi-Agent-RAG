export interface DocumentRecord {
  name: string
  path: string
  type: string
  pages?: number | string
  chunks: number
  chunk_size?: number | string
  chunk_overlap?: number | string
  uploaded_at?: string
  restored?: boolean
}

export interface Citation {
  source_number: number
  filename: string
  chunk_id: string
  text_preview: string
  score: number
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  workflow_metadata?: Record<string, unknown>
}

export interface TaskStatus {
  task_id: string
  type?: string
  status: 'pending' | 'running' | 'done' | 'error' | 'not_found'
  progress?: number
  current?: number
  total?: number
  stage?: string
  last_id?: string
  result?: any
  error?: string
  traceback?: string
}

export interface AppState {
  document_count: number
  message_count: number
  rag_initialized: boolean
  restore_status: string
  processing: boolean
  task_running: boolean
  eval_task_id: string | null
  ragas_evaluation_running: boolean
  neo4j: {
    available: boolean
    counts: { nodes: number; edges: number }
    top_entities: Array<[string, number]>
    error: string
  }
  performance: Record<string, number>
}

export interface EvaluationQuestion {
  id: string
  question: string
  question_type: string
  reference: string
  retrieved_chunk_ids?: string[]
  gold_chunk_ids?: string[]
}

export interface EvaluationScore {
  id: string
  question: string
  question_type?: string
  faithfulness?: number | null
  answer_relevancy?: number | null
  context_precision?: number | null
  context_recall?: number | null
  overall?: number | null
  retrieved_chunk_ids?: string[]
  gold_chunk_ids?: string[]
  error?: string | null
}

export interface EvaluationResult {
  run_id: string
  run_dir: string
  rag_outputs_path: string
  scores_path: string
  summary_path: string
  csv_path: string
  scores: EvaluationScore[]
  summary: Record<string, number | string | null>
  compatibility_warnings?: string[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    let message = response.statusText
    try {
      const data = await response.json()
      message = data.detail || data.error || message
    } catch {
      message = await response.text()
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),
  state: () => request<AppState>('/api/state'),
  documents: () => request<{ documents: DocumentRecord[] }>('/api/documents'),
  uploadDocument: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<{ task_id: string }>('/api/documents', { method: 'POST', body })
  },
  deleteDocument: (name: string) =>
    request<{ deleted: boolean; file_deleted: boolean; vector_chunks_deleted: boolean }>(
      `/api/documents/${encodeURIComponent(name)}`,
      { method: 'DELETE' }
    ),
  previewDocument: (name: string) =>
    request<{ name: string; text: string; chars: number; words: number }>(
      `/api/documents/${encodeURIComponent(name)}/preview`
    ),
  clearData: () => request<{ cleared: boolean }>('/api/data/clear', { method: 'POST' }),
  graphStats: () => request<any>('/api/graph/stats'),
  messages: () => request<{ messages: ChatMessage[] }>('/api/chat/messages'),
  ask: (query: string) =>
    request<ChatMessage>('/api/chat/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    }),
  startChatTask: (query: string) =>
    request<{ task_id: string }>('/api/chat/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    }),
  clearMessages: () => request<{ cleared: boolean }>('/api/chat/messages', { method: 'DELETE' }),
  exportChatUrl: () => '/api/chat/export',
  questions: (testFile = 'data/test_questions.json') =>
    request<{ test_file: string; questions: EvaluationQuestion[] }>(
      `/api/evaluation/questions?test_file=${encodeURIComponent(testFile)}`
    ),
  uploadQuestions: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<{ test_file: string; questions: EvaluationQuestion[] }>('/api/evaluation/questions', {
      method: 'POST',
      body
    })
  },
  startRagas: (test_file: string, reuse_rag_outputs: boolean) =>
    request<{ task_id: string }>('/api/evaluation/ragas', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_file, reuse_rag_outputs })
    }),
  task: (taskId: string) => request<TaskStatus>(`/api/tasks/${taskId}`),
  performance: () => request<Record<string, number>>('/api/performance'),
  savePerformance: () => request<{ path: string }>('/api/performance/save', { method: 'POST' })
}
