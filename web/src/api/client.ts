export interface DocumentRecord {
  name: string;
  path?: string;
  type?: string;
  pages?: number;
  chunks?: number;
  chunk_size?: number;
  chunk_overlap?: number;
  uploaded_at?: string;
  restored?: boolean;
}

export interface Citation {
  source_number: number;
  filename: string;
  chunk_id: string;
  text_preview: string;
  score: number;
}

export interface WorkflowMetadata {
  complexity?: number;
  strategy?: string;
  selected_retrievers?: string[];
  retriever_quotas?: Record<string, number>;
  retrieval_rounds?: number;
  validation_score?: number;
  critic_score?: number;
  regenerations?: number;
  decision?: string;
  reranker_used_cohere?: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  workflow_metadata?: WorkflowMetadata;
}

export interface Neo4jState {
  available: boolean;
  counts: {
    nodes?: number;
    edges?: number;
    relationships?: number;
  };
  top_entities: unknown[];
  error: string;
}

export interface PerformanceStats {
  total_queries?: number;
  avg_latency_ms?: number;
  min_latency_ms?: number;
  max_latency_ms?: number;
  cache_hit_rate?: number;
  cache_hits?: number;
  cache_misses?: number;
  cache_size?: number;
  avg_chunks?: number;
  session_duration_min?: number;
  [key: string]: unknown;
}

export interface SystemState {
  document_count: number;
  message_count: number;
  rag_initialized: boolean;
  restore_status: string;
  processing: boolean;
  task_running: boolean;
  eval_task_id: string | null;
  ragas_evaluation_running: boolean;
  neo4j: Neo4jState;
  performance: PerformanceStats;
}

export interface TaskState<T = unknown> {
  task_id: string;
  type?: string;
  status: "pending" | "running" | "done" | "error" | "not_found" | string;
  progress?: number;
  current?: number;
  total?: number;
  stage?: string;
  last_id?: string;
  error?: string;
  result?: T;
  submitted_at?: string;
  started_at?: string;
  finished_at?: string;
}

export interface PreviewDocument {
  name: string;
  text: string;
  chars: number;
  words: number;
}

export interface TestQuestion {
  id: string;
  question: string;
  question_type?: string;
  reference?: string;
}

export interface RagasResult {
  run_id?: string;
  output_dir?: string;
  jsonl_path?: string;
  csv_path?: string;
  summary_path?: string;
  summary?: Record<string, unknown>;
  scores?: Array<Record<string, unknown>>;
  compatibility_warnings?: string[];
  [key: string]: unknown;
}

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers }
  });

  if (!response.ok) {
    const message = await readError(response);
    throw new Error(message);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }

  return (await response.text()) as T;
}

async function readError(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const body = await response.json().catch(() => null);
    return String(body?.detail || body?.error || response.statusText);
  }

  const text = await response.text().catch(() => "");
  return text || response.statusText;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  state: () => request<SystemState>("/state"),
  graphStats: () => request<Neo4jState>("/graph/stats"),
  performance: () => request<PerformanceStats>("/performance"),
  savePerformance: () => request<{ status: string }>("/performance/save", { method: "POST" }),
  documents: () => request<{ documents: DocumentRecord[] }>("/documents"),
  uploadDocument: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ task_id: string }>("/documents", { method: "POST", body });
  },
  deleteDocument: (name: string) => request<Record<string, unknown>>(`/documents/${encodeURIComponent(name)}`, { method: "DELETE" }),
  previewDocument: (name: string) => request<PreviewDocument>(`/documents/${encodeURIComponent(name)}/preview`),
  clearData: () => request<{ cleared: boolean }>("/data/clear", { method: "POST" }),
  messages: () => request<{ messages: ChatMessage[] }>("/chat/messages"),
  createChatTask: (query: string) => request<{ task_id: string }>("/chat/tasks", { method: "POST", body: JSON.stringify({ query }) }),
  clearMessages: () => request<{ cleared: boolean }>("/chat/messages", { method: "DELETE" }),
  exportChat: () => request<string>("/chat/export"),
  evaluationQuestions: (testFile = "data/test_questions.json") => (
    request<{ test_file: string; questions: TestQuestion[] }>(`/evaluation/questions?test_file=${encodeURIComponent(testFile)}`)
  ),
  uploadEvaluationQuestions: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ test_file: string; questions: TestQuestion[] }>("/evaluation/questions", { method: "POST", body });
  },
  startRagas: (testFile: string, reuseRagOutputs: boolean) => (
    request<{ task_id: string }>("/evaluation/ragas", {
      method: "POST",
      body: JSON.stringify({ test_file: testFile, reuse_rag_outputs: reuseRagOutputs })
    })
  ),
  task: <T = unknown>(taskId: string) => request<TaskState<T>>(`/tasks/${encodeURIComponent(taskId)}`)
};
