"""后端依赖装配入口（Dependency Injection）。"""

from __future__ import annotations

from functools import lru_cache

from src.cache.redis_cache import RedisCacheService
from src.server.utils.state import RuntimeState
from src.server.utils.tasks import TaskRegistry
from src.use_cases.chat import (
    ClearMessagesUseCase,
    ExportChatHistoryUseCase,
    GetMessagesUseCase,
    RunChatQueryUseCase,
)
from src.use_cases.document import (
    ClearAllDataUseCase,
    DeleteDocumentUseCase,
    EnsurePersistedStateLoadedUseCase,
    ListDocumentsUseCase,
    PreviewDocumentUseCase,
    ProcessUploadedDocumentUseCase,
    RestoreDocumentStateUseCase,
    SaveUploadUseCase,
)
from src.use_cases.evaluation import (
    LoadTestQuestionsUseCase,
    SaveTestQuestionsUploadUseCase,
    SubmitRagasEvaluationUseCase,
)
from src.use_cases.system import (
    GetGraphStatsUseCase,
    GetPerformanceStatsUseCase,
    GetSystemStateUseCase,
    SavePerformanceMetricsUseCase,
)


@lru_cache(maxsize=1)
def get_runtime_state() -> RuntimeState:
    # 全局运行时状态（单例）
    return RuntimeState()


@lru_cache(maxsize=1)
def get_task_registry() -> TaskRegistry:
    # 后台任务注册中心（单例）
    return TaskRegistry()


@lru_cache(maxsize=1)
def get_cache_service() -> RedisCacheService:
    # Redis 缓存服务（单例）
    return RedisCacheService()


@lru_cache(maxsize=1)
def get_embedder():
    """延迟创建 Embedding 组件；缺少 openai 依赖时提供可控降级。"""
    try:
        from src.ingestion.embedder import EmbeddingGenerator
    except ModuleNotFoundError as exc:
        if exc.name != "openai":
            raise
        return _UnavailableEmbedder(exc)

    return EmbeddingGenerator(cache_service=get_cache_service())


@lru_cache(maxsize=1)
def get_run_chat_query_use_case() -> RunChatQueryUseCase:
    # 聊天主用例：负责完整问答流程编排
    return RunChatQueryUseCase(get_runtime_state(), get_task_registry(), get_embedder(), get_cache_service())


@lru_cache(maxsize=1)
def get_get_messages_use_case() -> GetMessagesUseCase:
    # 获取会话消息列表
    return GetMessagesUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_clear_messages_use_case() -> ClearMessagesUseCase:
    # 清空会话消息
    return ClearMessagesUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_export_chat_history_use_case() -> ExportChatHistoryUseCase:
    # 导出聊天记录
    return ExportChatHistoryUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_restore_document_state_use_case() -> RestoreDocumentStateUseCase:
    # 启动时恢复文档持久化状态
    return RestoreDocumentStateUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_ensure_persisted_state_loaded_use_case() -> EnsurePersistedStateLoadedUseCase:
    # 按需确保持久化状态已加载
    return EnsurePersistedStateLoadedUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_list_documents_use_case() -> ListDocumentsUseCase:
    # 文档列表查询
    return ListDocumentsUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_save_upload_use_case() -> SaveUploadUseCase:
    # 上传文件落盘
    return SaveUploadUseCase()


@lru_cache(maxsize=1)
def get_process_uploaded_document_use_case() -> ProcessUploadedDocumentUseCase:
    # 上传文档处理主链路（load/chunk/embed/store）
    return ProcessUploadedDocumentUseCase(get_runtime_state(), get_embedder(), get_cache_service())


@lru_cache(maxsize=1)
def get_delete_document_use_case() -> DeleteDocumentUseCase:
    # 删除单个文档及其关联索引
    return DeleteDocumentUseCase(get_runtime_state(), get_cache_service())


@lru_cache(maxsize=1)
def get_clear_all_data_use_case() -> ClearAllDataUseCase:
    # 清空所有文档与索引数据
    return ClearAllDataUseCase(get_runtime_state(), get_task_registry(), get_cache_service())


@lru_cache(maxsize=1)
def get_preview_document_use_case() -> PreviewDocumentUseCase:
    # 文档内容预览
    return PreviewDocumentUseCase(get_runtime_state(), get_list_documents_use_case())


@lru_cache(maxsize=1)
def get_load_test_questions_use_case() -> LoadTestQuestionsUseCase:
    # 读取评估问题
    return LoadTestQuestionsUseCase()


@lru_cache(maxsize=1)
def get_save_test_questions_upload_use_case() -> SaveTestQuestionsUploadUseCase:
    # 上传并保存评估问题文件
    return SaveTestQuestionsUploadUseCase(get_load_test_questions_use_case())


@lru_cache(maxsize=1)
def get_submit_ragas_evaluation_use_case() -> SubmitRagasEvaluationUseCase:
    # 提交 RAGAS 评估任务
    return SubmitRagasEvaluationUseCase(get_runtime_state(), get_task_registry(), get_embedder())


@lru_cache(maxsize=1)
def get_system_state_use_case() -> GetSystemStateUseCase:
    # 聚合系统状态（文档、任务、缓存等）
    return GetSystemStateUseCase(
        get_runtime_state(),
        get_task_registry(),
        get_ensure_persisted_state_loaded_use_case(),
        get_cache_service(),
    )


@lru_cache(maxsize=1)
def get_graph_stats_use_case() -> GetGraphStatsUseCase:
    # 图数据库统计信息
    return GetGraphStatsUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_performance_stats_use_case() -> GetPerformanceStatsUseCase:
    # 性能与缓存命中统计
    return GetPerformanceStatsUseCase(get_runtime_state(), get_cache_service())


@lru_cache(maxsize=1)
def get_save_performance_metrics_use_case() -> SavePerformanceMetricsUseCase:
    # 将性能统计落盘
    return SavePerformanceMetricsUseCase(get_runtime_state())


class _UnavailableEmbedder:
    # 当 openai 依赖缺失时，用于替代真实 embedder 的占位实现
    def __init__(self, exc: ModuleNotFoundError):
        self.exc = exc

    def get_stats(self) -> dict:
        return {
            "total_embeddings": 0,
            "model": "unavailable",
            "batch_size": 0,
            "embedding_dimension": 0,
        }

    def generate(self, texts):
        raise ModuleNotFoundError("openai package is required for embedding generation") from self.exc

    def generate_query_embedding(self, query):
        raise ModuleNotFoundError("openai package is required for embedding generation") from self.exc
