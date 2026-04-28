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
"""
鏄悗绔緷璧栬閰?渚濊禆娉ㄥ叆(DI)鐨勯泦涓叆鍙ｃ€?

瀹冭礋璐ｇ粺涓€鍒涘缓骞剁紦瀛樺簲鐢ㄨ繍琛屾墍闇€鐨勬牳蹇冨璞?渚嬪 RuntimeState銆?
TaskRegistry銆丷edisCacheService 浠ュ強鍚勭被 UseCase,閬垮厤鍦ㄦ瘡娆¤姹備腑
閲嶅鍒涘缓瀵硅薄,鍚屾椂鏂逛究鍦ㄤ笉鍚岃矾鐢卞拰鐢熷懡鍛ㄦ湡閽╁瓙涔嬮棿鍏变韩鐘舵€佷笌杩炴帴銆?

杩欓噷閫氳繃 @lru_cache(maxsize=1) 灏嗛儴鍒嗗璞″仛鎴愬綋鍓嶈繘绋嬪唴鐨勫崟渚嬨€?
绗竴娆¤皟鐢?get_* 鍑芥暟鏃朵細鍒涘缓瀵硅薄,鍚庣画璋冪敤浼氱洿鎺ュ鐢ㄥ凡鍒涘缓鐨勫疄渚嬨€?

鍚勭被 UseCase 鐨勪緷璧栧叧绯讳篃鍦ㄨ繖閲岀粺涓€瑁呴厤,渚嬪 RunChatQueryUseCase
浼氫緷璧?RuntimeState銆乀askRegistry銆丒mbedder 鍜?CacheService銆?
璺敱灞傚彧闇€瑕佽皟鐢ㄥ搴旂殑 get_*_use_case(),涓嶉渶瑕佸叧蹇冨璞″浣曞垱寤恒€?
"""


@lru_cache(maxsize=1)#缂撳瓨
def get_runtime_state() -> RuntimeState:
    # 鍒涘缓骞剁紦瀛樺叏灞€杩愯鏃剁姸鎬佸疄渚?
    return RuntimeState()


@lru_cache(maxsize=1)
def get_task_registry() -> TaskRegistry:
    # 鍒涘缓骞剁紦瀛樹换鍔℃敞鍐岃〃瀹炰緥
    return TaskRegistry()


@lru_cache(maxsize=1)
def get_cache_service() -> RedisCacheService:
    # 鍒涘缓骞剁紦瀛?Redis 缂撳瓨鏈嶅姟瀹炰緥
    return RedisCacheService()


@lru_cache(maxsize=1)
def get_embedder():
    # 灏濊瘯寤惰繜瀵煎叆宓屽叆鐢熸垚鍣?閬垮厤鍚姩鏃跺己渚濊禆 OpenAI 鍖?
    try:
        from src.ingestion.embedder import EmbeddingGenerator
    except ModuleNotFoundError as exc:
        # 濡傛灉缂哄け鐨勪笉鏄?openai 鍖?鍒欑户缁姏鍑哄師濮嬪紓甯?
        if exc.name != "openai":
            raise

        # openai 鍖呬笉鍙敤鏃惰繑鍥炲崰浣嶅祵鍏ュ櫒
        return _UnavailableEmbedder(exc)

    # 鍒涘缓骞剁紦瀛樺祵鍏ョ敓鎴愬櫒瀹炰緥
    return EmbeddingGenerator(cache_service=get_cache_service())


@lru_cache(maxsize=1)
def get_run_chat_query_use_case() -> RunChatQueryUseCase:
    # 鍒涘缓骞剁紦瀛樿亰澶╂煡璇㈢敤渚?
    return RunChatQueryUseCase(get_runtime_state(), get_task_registry(), get_embedder(), get_cache_service())


@lru_cache(maxsize=1)
def get_get_messages_use_case() -> GetMessagesUseCase:
    # 鍒涘缓骞剁紦瀛樿幏鍙栨秷鎭敤渚?
    return GetMessagesUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_clear_messages_use_case() -> ClearMessagesUseCase:
    # 鍒涘缓骞剁紦瀛樻竻绌烘秷鎭敤渚?
    return ClearMessagesUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_export_chat_history_use_case() -> ExportChatHistoryUseCase:
    # 鍒涘缓骞剁紦瀛樺鍑鸿亰澶╁巻鍙茬敤渚?
    return ExportChatHistoryUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_restore_document_state_use_case() -> RestoreDocumentStateUseCase:
    # 鍒涘缓骞剁紦瀛樻仮澶嶆枃妗ｇ姸鎬佺敤渚?
    return RestoreDocumentStateUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_ensure_persisted_state_loaded_use_case() -> EnsurePersistedStateLoadedUseCase:
    # 鍒涘缓骞剁紦瀛樼‘淇濇寔涔呭寲鐘舵€佸凡鍔犺浇鐨勭敤渚?
    return EnsurePersistedStateLoadedUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_list_documents_use_case() -> ListDocumentsUseCase:
    # 鍒涘缓骞剁紦瀛樻枃妗ｅ垪琛ㄦ煡璇㈢敤渚?
    return ListDocumentsUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_save_upload_use_case() -> SaveUploadUseCase:
    # 鍒涘缓骞剁紦瀛樹笂浼犳枃浠朵繚瀛樼敤渚?
    return SaveUploadUseCase()


@lru_cache(maxsize=1)
def get_process_uploaded_document_use_case() -> ProcessUploadedDocumentUseCase:
    # 鍒涘缓骞剁紦瀛樹笂浼犳枃妗ｅ鐞嗙敤渚?
    return ProcessUploadedDocumentUseCase(get_runtime_state(), get_embedder(), get_cache_service())


@lru_cache(maxsize=1)
def get_delete_document_use_case() -> DeleteDocumentUseCase:
    # 鍒涘缓骞剁紦瀛樻枃妗ｅ垹闄ょ敤渚?
    return DeleteDocumentUseCase(get_runtime_state(), get_cache_service())


@lru_cache(maxsize=1)
def get_clear_all_data_use_case() -> ClearAllDataUseCase:
    # 鍒涘缓骞剁紦瀛樻竻绌哄叏閮ㄦ暟鎹敤渚?
    return ClearAllDataUseCase(get_runtime_state(), get_task_registry(), get_cache_service())


@lru_cache(maxsize=1)
def get_preview_document_use_case() -> PreviewDocumentUseCase:
    # 鍒涘缓骞剁紦瀛樻枃妗ｉ瑙堢敤渚?
    return PreviewDocumentUseCase(get_runtime_state(), get_list_documents_use_case())


@lru_cache(maxsize=1)
def get_load_test_questions_use_case() -> LoadTestQuestionsUseCase:
    # 鍒涘缓骞剁紦瀛樻祴璇曢棶棰樺姞杞界敤渚?
    return LoadTestQuestionsUseCase()


@lru_cache(maxsize=1)
def get_save_test_questions_upload_use_case() -> SaveTestQuestionsUploadUseCase:
    # 鍒涘缓骞剁紦瀛樻祴璇曢棶棰樹笂浼犱繚瀛樼敤渚?
    return SaveTestQuestionsUploadUseCase(get_load_test_questions_use_case())


@lru_cache(maxsize=1)
def get_submit_ragas_evaluation_use_case() -> SubmitRagasEvaluationUseCase:
    # 鍒涘缓骞剁紦瀛?RAGAS 璇勪及鎻愪氦鐢ㄤ緥
    return SubmitRagasEvaluationUseCase(get_runtime_state(), get_task_registry(), get_embedder())


@lru_cache(maxsize=1)
def get_system_state_use_case() -> GetSystemStateUseCase:
    # 鍒涘缓骞剁紦瀛樼郴缁熺姸鎬佹煡璇㈢敤渚?
    return GetSystemStateUseCase(
        get_runtime_state(),
        get_task_registry(),
        get_ensure_persisted_state_loaded_use_case(),
        get_cache_service(),
    )


@lru_cache(maxsize=1)
def get_graph_stats_use_case() -> GetGraphStatsUseCase:
    # 鍒涘缓骞剁紦瀛樺浘鏁版嵁缁熻鐢ㄤ緥
    return GetGraphStatsUseCase(get_runtime_state())


@lru_cache(maxsize=1)
def get_performance_stats_use_case() -> GetPerformanceStatsUseCase:
    # 鍒涘缓骞剁紦瀛樻€ц兘缁熻鏌ヨ鐢ㄤ緥
    return GetPerformanceStatsUseCase(get_runtime_state(), get_cache_service())


@lru_cache(maxsize=1)
def get_save_performance_metrics_use_case() -> SavePerformanceMetricsUseCase:
    # 鍒涘缓骞剁紦瀛樻€ц兘鎸囨爣淇濆瓨鐢ㄤ緥
    return SavePerformanceMetricsUseCase(get_runtime_state())


class _UnavailableEmbedder:
    # 瀹氫箟 OpenAI 鍖呬笉鍙敤鏃朵娇鐢ㄧ殑鍗犱綅宓屽叆鍣?
    def __init__(self, exc: ModuleNotFoundError):
        # 淇濆瓨鍘熷寮傚父,渚夸簬鍚庣画鎶涘嚭鏃朵繚鐣欎笂涓嬫枃
        self.exc = exc

    def get_stats(self) -> dict:
        # 杩斿洖涓嶅彲鐢ㄧ姸鎬佷笅鐨勯粯璁ゅ祵鍏ョ粺璁′俊鎭?
        return {"total_embeddings": 0, "model": "unavailable", "batch_size": 0, "embedding_dimension": 0}

    def generate(self, texts):
        # 鐢熸垚鏂囨。宓屽叆鏃舵彁绀虹己灏?openai 渚濊禆
        raise ModuleNotFoundError("openai package is required for embedding generation") from self.exc

    def generate_query_embedding(self, query):
        # 鐢熸垚鏌ヨ宓屽叆鏃舵彁绀虹己灏?openai 渚濊禆
        raise ModuleNotFoundError("openai package is required for embedding generation") from self.exc

