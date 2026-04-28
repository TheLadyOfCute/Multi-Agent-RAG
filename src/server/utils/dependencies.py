"""Dependency wiring for the backend application."""

from __future__ import annotations

from functools import lru_cache

from src.server.utils.state import RuntimeState
from src.server.utils.tasks import TaskRegistry
from src.cache.redis_cache import RedisCacheService
from src.use_cases.chat import (
    ClearMessagesUseCase,
    ExportChatHistoryUseCase,
    GetMessagesUseCase,
    RunChatQueryUseCase,
)
from src.use_cases.documents import (
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
"""
是后端依赖装配/依赖注入(DI)的集中入口。

它负责统一创建并缓存应用运行所需的核心对象,例如 RuntimeState、
TaskRegistry、RedisCacheService 以及各类 UseCase,避免在每次请求中
重复创建对象,同时方便在不同路由和生命周期钩子之间共享状态与连接。

这里通过 @lru_cache(maxsize=1) 将部分对象做成当前进程内的单例。
第一次调用 get_* 函数时会创建对象,后续调用会直接复用已创建的实例。

各类 UseCase 的依赖关系也在这里统一装配,例如 RunChatQueryUseCase
会依赖 RuntimeState、TaskRegistry、Embedder 和 CacheService。
路由层只需要调用对应的 get_*_use_case(),不需要关心对象如何创建。
"""


@lru_cache(maxsize=1)#缓存
def get_runtime_state() -> RuntimeState:
    # 创建并缓存全局运行时状态实例
    return RuntimeState()


@lru_cache(maxsize=1)
def get_task_registry() -> TaskRegistry:
    # 创建并缓存任务注册表实例
    return TaskRegistry()


@lru_cache(maxsize=1)
def get_cache_service() -> RedisCacheService:
    # 创建并缓存 Redis 缓存服务实例
    return RedisCacheService()


@lru_cache(maxsize=1)
def get_embedder():
    # 尝试延迟导入嵌入生成器,避免启动时强依赖 OpenAI 包
    try:
        from src.ingestion.embedder import EmbeddingGenerator
    except ModuleNotFoundError as exc:
        # 如果缺失的不是 openai 包,则继续抛出原始异常
        if exc.name != "openai":
            raise

        # openai 包不可用时返回占位嵌入器
        return _UnavailableEmbedder(exc)

    # 创建并缓存嵌入生成器实例
    return EmbeddingGenerator(cache_service=get_cache_service())


@lru_cache(maxsize=1)
def get_run_chat_query_use_case() -> RunChatQueryUseCase:
    # 创建并缓存聊天查询用例
    return RunChatQueryUseCase(get_runtime_state(), get_task_registry(), get_embedder(), get_cache_service())


@lru_cache(maxsize=1)
def get_get_messages_use_case() -> GetMessagesUseCase:
    # 创建并缓存获取消息用例
    return GetMessagesUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_clear_messages_use_case() -> ClearMessagesUseCase:
    # 创建并缓存清空消息用例
    return ClearMessagesUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_export_chat_history_use_case() -> ExportChatHistoryUseCase:
    # 创建并缓存导出聊天历史用例
    return ExportChatHistoryUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_restore_document_state_use_case() -> RestoreDocumentStateUseCase:
    # 创建并缓存恢复文档状态用例
    return RestoreDocumentStateUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_ensure_persisted_state_loaded_use_case() -> EnsurePersistedStateLoadedUseCase:
    # 创建并缓存确保持久化状态已加载的用例
    return EnsurePersistedStateLoadedUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_list_documents_use_case() -> ListDocumentsUseCase:
    # 创建并缓存文档列表查询用例
    return ListDocumentsUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_save_upload_use_case() -> SaveUploadUseCase:
    # 创建并缓存上传文件保存用例
    return SaveUploadUseCase()


@lru_cache(maxsize=1)
def get_process_uploaded_document_use_case() -> ProcessUploadedDocumentUseCase:
    # 创建并缓存上传文档处理用例
    return ProcessUploadedDocumentUseCase(get_runtime_state(), get_embedder(), get_cache_service())


@lru_cache(maxsize=1)
def get_delete_document_use_case() -> DeleteDocumentUseCase:
    # 创建并缓存文档删除用例
    return DeleteDocumentUseCase(get_runtime_state(), get_cache_service())


@lru_cache(maxsize=1)
def get_clear_all_data_use_case() -> ClearAllDataUseCase:
    # 创建并缓存清空全部数据用例
    return ClearAllDataUseCase(get_runtime_state(), get_task_registry(), get_cache_service())


@lru_cache(maxsize=1)
def get_preview_document_use_case() -> PreviewDocumentUseCase:
    # 创建并缓存文档预览用例
    return PreviewDocumentUseCase(get_runtime_state(), get_list_documents_use_case())


@lru_cache(maxsize=1)
def get_load_test_questions_use_case() -> LoadTestQuestionsUseCase:
    # 创建并缓存测试问题加载用例
    return LoadTestQuestionsUseCase()


@lru_cache(maxsize=1)
def get_save_test_questions_upload_use_case() -> SaveTestQuestionsUploadUseCase:
    # 创建并缓存测试问题上传保存用例
    return SaveTestQuestionsUploadUseCase(get_load_test_questions_use_case())


@lru_cache(maxsize=1)
def get_submit_ragas_evaluation_use_case() -> SubmitRagasEvaluationUseCase:
    # 创建并缓存 RAGAS 评估提交用例
    return SubmitRagasEvaluationUseCase(get_runtime_state(), get_task_registry(), get_embedder())


@lru_cache(maxsize=1)
def get_system_state_use_case() -> GetSystemStateUseCase:
    # 创建并缓存系统状态查询用例
    return GetSystemStateUseCase(
        get_runtime_state(),
        get_task_registry(),
        get_ensure_persisted_state_loaded_use_case(),
        get_cache_service(),
    )


@lru_cache(maxsize=1)
def get_graph_stats_use_case() -> GetGraphStatsUseCase:
    # 创建并缓存图数据统计用例
    return GetGraphStatsUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_performance_stats_use_case() -> GetPerformanceStatsUseCase:
    # 创建并缓存性能统计查询用例
    return GetPerformanceStatsUseCase(get_runtime_state(), get_cache_service())


@lru_cache(maxsize=1)
def get_save_performance_metrics_use_case() -> SavePerformanceMetricsUseCase:
    # 创建并缓存性能指标保存用例
    return SavePerformanceMetricsUseCase(get_runtime_state())


class _UnavailableEmbedder:
    # 定义 OpenAI 包不可用时使用的占位嵌入器
    def __init__(self, exc: ModuleNotFoundError):
        # 保存原始异常,便于后续抛出时保留上下文
        self.exc = exc

    def get_stats(self) -> dict:
        # 返回不可用状态下的默认嵌入统计信息
        return {"total_embeddings": 0, "model": "unavailable", "batch_size": 0, "embedding_dimension": 0}

    def generate(self, texts):
        # 生成文档嵌入时提示缺少 openai 依赖
        raise ModuleNotFoundError("openai package is required for embedding generation") from self.exc

    def generate_query_embedding(self, query):
        # 生成查询嵌入时提示缺少 openai 依赖
        raise ModuleNotFoundError("openai package is required for embedding generation") from self.exc