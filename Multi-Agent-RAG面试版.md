# Multi-Agent RAG 项目面试版

> 用途：这份文档分为两部分。第一部分可以直接复制到简历；第二部分用于面试前准备，覆盖项目介绍、架构设计、RAG 检索、Multi-Agent 编排、知识图谱、缓存、评估、API 与工程化等高频追问。

## 一、简历可复制版本

### Multi-Agent RAG 智能文档问答系统

**项目简介：**  
基于 Multi-Agent RAG 架构实现的智能文档问答系统，支持 PDF/DOCX/TXT 文档上传、自动切分、向量化入库、混合检索、答案引用溯源、异步任务进度展示和 RAGAS 自动评估。系统通过 LangGraph 编排多个 Agent，将传统“检索 + 生成”流程拆分为问题拆解、检索规划、多路召回、重排、证据校验、答案生成和质量审查等环节，提升复杂问题场景下的召回完整性、答案可靠性和系统可观测性。

**技术栈：**  
FastAPI、Vue 3、TypeScript、LangGraph、ChromaDB、Neo4j、Redis、BM25、Docker

**主要功能：**

- **文档处理与索引构建：** 支持 PDF/DOCX/TXT 文档上传与预览，完成文档解析、动态 chunk 参数选择、文本切分、批量 Embedding 生成，并将文档块持久化写入 ChromaDB。
- **Multi-Agent 问答工作流：** 基于 LangGraph 构建 7 节点 Agent 流程，包括 `QueryDecomposer`、`Planner`、`RetrievalCoordinator`、`Reranker`、`Validator`、`Writer`、`Critic`，实现问题拆解、检索规划、证据校验和答案质量反思。
- **混合检索增强：** 实现向量检索、BM25 关键词检索和 Neo4j 知识图谱检索的多路召回；Planner 可根据子问题动态选择检索器和 top_k 配额，覆盖语义相似、精确术语和实体关系类问题。
- **答案可靠性控制：** Validator 根据相关性、覆盖度和置信度判断证据是否充分，证据不足时触发二次检索；Critic 从准确性、完整性、引用、清晰度和相关性维度评估答案，必要时驱动重新生成。
- **缓存与状态一致性：** 接入 Redis 缓存 Embedding、Query Embedding 和问答结果，并基于知识库状态 token 避免文档更新后命中旧答案；上传、删除或清空文档后自动清理答案缓存并同步更新索引。
- **自动化评估与工程化：** 接入 RAGAS 批量评估，输出 `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall` 等指标；采用 FastAPI 分层架构、异步任务注册、持久化恢复和 Pytest 测试覆盖核心链路。

### 面试 30 秒项目介绍

这是一个智能文档问答系统，核心不是简单地把文档放进向量库再调用大模型，而是做了一个 Agentic RAG 流程。用户上传 PDF、DOCX 或 TXT 后，系统会自动读取、切分、生成 Embedding，并同时写入 ChromaDB、BM25 索引和 Neo4j 知识图谱。用户提问时，LangGraph 会编排多个 Agent：先判断是否需要拆解问题，再为每个子问题规划向量、关键词或图谱检索，之后融合和重排证据，由 Validator 判断证据是否足够，不够就触发二次检索；最后 Writer 生成带引用的答案，Critic 再检查准确性、完整性、引用和清晰度，必要时让 Writer 根据反馈重写。系统还接了 Redis 缓存和 RAGAS 自动评估，能从性能和质量两个角度闭环优化。

### 面试 1 分钟项目介绍

我做的是一个 Multi-Agent RAG 智能文档问答系统，技术栈是 FastAPI、Vue 3、LangGraph、ChromaDB、Neo4j、Redis 和 RAGAS。系统支持用户上传 PDF、DOCX、TXT 文档，后端完成文档解析、动态 chunk 参数选择、文本切分、批量 Embedding、向量库持久化，同时构建 BM25 关键词索引和 Neo4j 实体关系图谱。

问答链路上，我用 LangGraph 搭了 7 个节点的 Agent 工作流，包括问题拆解、检索规划、多路检索协调、重排、证据校验、答案生成和质量审查。它的关键点是 Planner 会对每个子问题动态选择检索器，比如语义问题走向量检索，精确术语走 BM25，实体关系问题走图谱检索。Validator 会根据相关性、覆盖度和置信度判断证据是否足够，不足时会扩大检索范围重试；Critic 会对生成答案从准确性、完整性、引用、清晰度和相关性打分，低于阈值就触发重写。

工程上，我做了 FastAPI 分层接口、异步任务进度、Redis 缓存、知识库状态 token、文档删除后的索引同步、启动时持久化恢复，以及 RAGAS 批量评估，最终可以用 faithfulness、answer relevancy、context precision、context recall 等指标衡量系统质量。

## 二、面试高频问答

### 1. 项目整体

**Q1：这个项目一句话怎么介绍？**  
A：这是一个基于 Multi-Agent 的 RAG 文档问答系统，支持文档上传、混合检索、知识图谱增强、答案引用、缓存加速和 RAGAS 自动评估。相比普通 RAG，它把问题拆解、检索规划、证据校验和答案质量审查都做成了可观测的 Agent 节点。

**Q2：这个项目解决了什么问题？**  
A：解决企业或个人知识库中文档内容难检索、答案不可溯源、复杂问题单次向量检索召回不充分、生成质量难评估的问题。系统通过多路检索提高召回，通过引用和 Validator 降低幻觉，通过 RAGAS 做质量量化。

**Q3：你的核心贡献是什么？**  
A：核心贡献有四个：第一，设计 LangGraph 多 Agent 工作流；第二，实现向量、BM25、图谱三路混合检索和动态检索规划；第三，实现 Validator 重试和 Critic 再生成机制；第四，接入 Redis 缓存和 RAGAS 评估，把系统从 Demo 做成可运行、可调试、可评估的服务。

**Q4：为什么叫 Multi-Agent RAG？**  
A：因为系统不是一个单一链路，而是由多个职责明确的 Agent 协作完成：QueryDecomposer 拆问题，Planner 规划检索策略，RetrievalCoordinator 调度检索器，Reranker 重排，Validator 判断证据是否足够，Writer 生成答案，Critic 审查答案质量。

**Q5：和普通 RAG 最大区别是什么？**  
A：普通 RAG 通常是“向量检索 top_k -> 拼上下文 -> LLM 回答”。这个项目增加了问题拆解、动态检索器选择、混合召回、证据充分性校验、失败重试、答案质量反思和自动评估，所以更适合复杂问题和需要可解释调试的场景。

**Q6：项目整体链路是什么？**  
A：文档链路是：上传文档 -> 解析文本 -> 选择 chunk 参数 -> 切分 -> 生成 Embedding -> 写入 ChromaDB -> 构建 BM25 -> 抽取实体关系写入 Neo4j。问答链路是：用户提问 -> 查缓存 -> Multi-Agent 工作流 -> 三路检索 -> 重排和验证 -> 生成答案与引用 -> Critic 检查 -> 写缓存和会话记录。

### 2. 架构设计

**Q7：后端为什么采用 routes -> use cases -> infrastructure 的分层？**  
A：routes 只负责 HTTP 请求响应，use cases 负责业务流程编排，底层模块负责存储、检索、缓存、评估等基础能力。这样接口层和业务逻辑不会混在一起，后续替换向量库、检索器或前端都比较容易。

**Q8：项目主要模块怎么划分？**  
A：`server` 是 FastAPI 入口和路由，`use_cases` 是应用业务，`agents` 是 Agent 实现，`workflows` 是 LangGraph 编排，`retrieval` 是向量/BM25/图谱检索，`ingestion` 是文档加载切分和 Embedding，`storage` 是 ChromaDB，`graph` 是 Neo4j 图谱，`cache` 是 Redis，`evaluation` 是 RAGAS。

**Q9：为什么选择 FastAPI？**  
A：FastAPI 原生支持异步、Pydantic 数据校验和自动 API 文档，适合做前后端分离的 AI 服务接口。项目里文档上传、异步任务、聊天接口、评估接口都比较适合用 FastAPI 管理。

**Q10：为什么用 LangGraph 而不是普通 LangChain Chain？**  
A：因为这个流程有条件分支和循环，比如 Validator 不通过要回到 retrieval，Critic 不通过要回到 writer。LangGraph 的 StateGraph 更适合表达有状态、多节点、可重试的工作流，而普通链式调用表达这种流程会比较硬。

**Q11：AgentState 里保存了什么？**  
A：保存了整个工作流共享状态，包括原始 query、问题复杂度、策略、子问题、子问题检索计划、检索到的 chunks、检索轮次、验证分数、答案、Critic 分数和反馈、以及各节点 metadata。

**Q12：为什么要保留 workflow metadata？**  
A：metadata 可以告诉前端和开发者本次问答用了哪些检索器、每个检索器配额是多少、检索了几轮、Validator 和 Critic 分数是多少、是否使用了重排。这对调试 RAG 效果非常关键。

### 3. 文档入库

**Q13：文档上传后做了哪些处理？**  
A：保存上传文件后，系统读取文档文本，调用 ChunkingAdvisor 选择 chunk_size 和 overlap，使用 FlatChunker 切分，批量生成 Embedding，然后写入 ChromaDB。同时还会抽取实体关系写入 Neo4j，并基于向量库内容重建 BM25 索引。

**Q14：支持哪些文档格式？**  
A：支持 PDF、DOCX 和 TXT。PDF 使用 pypdf，DOCX 使用 python-docx，TXT 直接读取文本。

**Q15：为什么要切分文档？**  
A：因为 LLM 上下文有限，向量检索也需要比较小粒度的语义单元。切分后可以提高检索命中精度，也能让答案引用定位到更具体的片段。

**Q16：chunk_size 和 overlap 怎么定？**  
A：项目里优先使用 ChunkingAdvisorAgent 根据文档内容建议参数，失败时回退到配置里的默认值，比如 chunk_size=500、overlap=50。overlap 用来保留跨 chunk 的上下文连续性。

**Q17：为什么上传文档后要清理答案缓存？**  
A：因为知识库内容变了，同一个问题的正确答案可能变化。如果继续复用旧答案，会出现知识库已更新但回答仍基于旧文档的问题，所以上传、删除、清空文档后要清理答案缓存。

**Q18：删除文档时如何保证索引一致？**  
A：删除文档会删除上传文件、删除 ChromaDB 中对应 chunks、删除 Neo4j 中对应 chunk 证据，并根据剩余向量数据重建 BM25。如果没有剩余 chunk，就删除 BM25 索引并把 RAG 状态标记为未初始化。

### 4. 检索设计

**Q19：为什么要做混合检索？**  
A：不同问题适合不同检索方式。向量检索适合语义相似和概念类问题；BM25 适合型号、编号、专有名词等精确匹配；图谱检索适合实体关系、多跳路径和因果链路问题。混合检索能提升复杂问题的召回率。

**Q20：三种检索分别适合什么场景？**  
A：向量检索适合“是什么、为什么、如何理解”这类语义问题；BM25 适合“某个精确术语、错误码、标题、版本号在哪里”；图谱检索适合“谁和谁有什么关系、A 如何影响 B、多个概念之间链路是什么”。

**Q21：Planner 怎么选择检索器？**  
A：QueryDecomposer 先生成子问题，Planner 再让 LLM 对每个子问题输出 JSON 检索计划，包括 query、retrievers 和 quotas。系统会校验 retriever 是否属于 vector、keyword、graph，并限制 top_k 配额，解析失败时使用规则兜底。

**Q22：如果 LLM 输出的检索计划格式不对怎么办？**  
A：Planner 会清理 markdown 代码块，提取 JSON 数组并做字段校验。如果无法解析或某条计划非法，就回退到规则策略，比如关系类问题选 graph+vector，精确匹配选 keyword，普通语义问题选 vector。

**Q23：RetrievalCoordinator 做什么？**  
A：它根据每个子问题的检索计划调用对应检索器，收集所有候选 chunks，然后按 chunk_id 或内容去重，保留分数更高的版本，并合并命中的检索来源，比如一个 chunk 同时被 vector 和 keyword 命中。

**Q24：为什么要去重？**  
A：混合检索会让同一 chunk 从不同路径重复出现。如果不去重，LLM 上下文会被重复证据占用，降低信息密度，也会影响引用和重排结果。

**Q25：Reranker 的作用是什么？**  
A：Reranker 对多路召回后的候选 chunk 进行二次排序，把更可能回答问题的证据排到前面。项目支持 Cohere rerank，无法使用时有加权 fallback。

**Q26：检索 top_k 越大越好吗？**  
A：不是。top_k 太小容易漏召回，太大则会引入噪声、增加上下文长度和 LLM 成本。项目用 Planner 为不同检索器分配配额，并通过 Validator 检查是否需要扩大检索。

**Q27：如何处理复杂问题？**  
A：先用 QueryDecomposer 判断是否需要拆解。如果是多方面、多跳或组合问题，就拆成多个子问题；Planner 对每个子问题单独选择检索器；最后 RetrievalCoordinator 聚合所有子问题证据。

### 5. 知识图谱

**Q28：为什么引入 Neo4j 知识图谱？**  
A：向量检索擅长语义相似，但对实体关系和多跳路径不够直观。Neo4j 可以把文档中的实体和关系结构化，支持按实体路径、邻居扩展和关系证据召回，适合关系型问题。

**Q29：知识图谱怎么构建？**  
A：文档切分后，对每个 chunk 做实体抽取；当 chunk 中有两个或以上实体时，关系抽取器会抽取实体间关系；然后把 chunk、实体和关系写入 Neo4j，保留 chunk_id 作为证据回溯。

**Q30：图谱检索逻辑是什么？**  
A：先从 query 中抽取实体并过滤出图中存在的实体。若有多个实体，就查找它们之间 3 跳以内的路径，按路径长度、关系置信度和具体关系类型排序，再根据路径实体召回相关 chunk。若没有路径，则做实体邻居扩展兜底。

**Q31：单实体问题怎么做图检索？**  
A：单实体时先做一跳邻居扩展，根据扩展实体召回 chunk；如果没有结果，再尝试两跳邻居扩展。这样能避免只匹配实体本身导致证据过少。

**Q32：图谱检索怎么给 chunk 排序？**  
A：会统计 chunk 中覆盖了多少路径实体，并结合路径分数给 bonus。路径越短、关系置信度越高、关系类型越具体，相关 chunk 分数越高。

**Q33：图谱检索失败会影响问答吗？**  
A：不会。图谱构建和图检索都是 best-effort，失败时系统会记录 warning 并继续走向量和 BM25 检索，保证主链路可用。

### 6. Validator 与 Critic

**Q34：Validator 和 Critic 有什么区别？**  
A：Validator 在生成前判断“证据够不够”，关注 chunks 是否能回答问题；Critic 在生成后判断“答案好不好”，关注准确性、完整性、引用、清晰度和相关性。

**Q35：Validator 怎么打分？**  
A：Validator 综合三类分数：相关性占 50%，覆盖度占 30%，置信度占 20%。相关性主要由 LLM 判断，覆盖度考虑 chunk 数量和来源多样性，置信度考虑 chunk 分数均值、最低分和一致性。

**Q36：Validator 不通过怎么办？**  
A：如果验证分数低于阈值并且没达到最大重试次数，会返回 `RETRIEVE_MORE`。LangGraph 条件边会让流程回到 retrieval，并强制下一轮使用 vector、keyword、graph 三种检索器扩大召回。

**Q37：为什么最大重试次数有限制？**  
A：避免在检索不足时无限循环，也控制延迟和成本。达到最大重试后系统会强制进入生成阶段，但 metadata 会保留较低的 validation_score，方便后续排查。

**Q38：Critic 怎么判断答案质量？**  
A：Critic 使用 LLM 按 accuracy、completeness、citations、clarity、relevance 五个维度输出 JSON 分数，再按权重计算 overall_score。低于阈值则让 Writer 根据反馈重新生成。

**Q39：Critic 会不会导致成本上升？**  
A：会增加一次或多次 LLM 调用，所以项目设置了 max_iterations。它适合对准确性要求较高的文档问答；如果追求低延迟，可以降低 Critic 次数或只在复杂问题上启用。

**Q40：如何降低 hallucination？**  
A：主要靠四点：多路检索提高证据召回，Validator 在生成前检查证据充分性，Writer 基于 chunk 生成并输出引用，Critic 在生成后检查准确性和引用质量。同时用 RAGAS 的 faithfulness 做离线评估。

### 7. 缓存与性能

**Q41：Redis 缓存了什么？**  
A：缓存了文档 chunk embedding、query embedding、问答结果和性能统计。Embedding 缓存减少重复向量化成本，答案缓存减少重复问题的端到端延迟。

**Q42：答案缓存 key 怎么避免脏读？**  
A：答案缓存不仅和 query 有关，还和知识库状态 token 有关。这个 token 由当前文档快照构建，文档变化后 token 变化，旧答案自然不会命中。

**Q43：Redis 不可用怎么办？**  
A：系统会记录 warning 并跳过缓存，不影响主流程运行。这是因为缓存是性能优化，不应该成为问答链路的强依赖。

**Q44：你做了哪些性能优化？**  
A：批量生成 Embedding、缓存 Embedding 和答案、BM25 索引持久化、启动时从 Chroma 恢复状态、异步任务进度避免长请求阻塞前端体验，以及只在需要时打开向量库和图数据库连接。

**Q45：为什么要异步任务？**  
A：文档上传、图谱构建和 RAGAS 评估耗时较长。异步任务可以先返回 task_id，前端轮询进度，用户体验更好，也避免 HTTP 请求长时间无响应。

### 8. RAGAS 评估

**Q46：为什么要接 RAGAS？**  
A：RAG 系统不能只靠人工感觉判断好坏。RAGAS 可以从忠实度、答案相关性、上下文精确率和召回率等维度量化质量，方便比较不同检索策略或参数调整的效果。

**Q47：项目评估哪些指标？**  
A：主要评估 `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`，并计算 overall 平均分。

**Q48：faithfulness 是什么？**  
A：回答是否忠实于检索到的上下文。分数低通常说明答案有幻觉，或者回答引入了上下文没有支持的信息。

**Q49：context_precision 和 context_recall 分别代表什么？**  
A：context_precision 看检索到的上下文中有多少是有用证据，反映噪声大小；context_recall 看答案所需证据是否被召回，反映漏召回情况。

**Q50：RAGAS 评估流程是什么？**  
A：加载测试问题和参考答案，逐条调用完整 RAG 工作流生成答案和上下文，RagasEvaluationAgent 整理成评估输入，RAGASEvaluator 调用 RAGAS 计算指标，最后输出 JSONL、CSV 和 summary。

**Q51：如果 RAGAS 分数不高，你会怎么排查？**  
A：先看 context_recall，如果低说明检索漏了，要调 chunk、top_k、检索器选择或图谱；再看 context_precision，如果低说明噪声多，要调重排和过滤；如果 faithfulness 低，重点优化 Writer prompt、引用约束和 Critic；如果 answer_relevancy 低，要看问题拆解和生成是否偏题。

### 9. API 与前端

**Q52：主要接口有哪些？**  
A：文档接口包括上传、列表、删除和预览；聊天接口包括同步问答、异步问答任务、消息列表、清空和导出；评估接口包括加载测试问题、上传测试问题和提交 RAGAS 评估；系统接口包括健康检查、统计和任务查询。

**Q53：同步问答和异步问答有什么区别？**  
A：同步接口适合调试，直接等待完整回答；异步接口适合前端正式使用，先返回 task_id，然后通过任务接口轮询进度和结果。

**Q54：前端做了什么？**  
A：前端基于 Vue 3 + Vite + TypeScript，提供文档上传删除和预览、聊天问答、引用展开、任务进度、RAGAS 评估结果展示、系统统计和对话导出。

**Q55：引用怎么实现？**  
A：Writer 的 metadata 会记录引用编号和 chunk_id。后端根据返回的 chunks 和 citation_ids 格式化 citations，包含 source_number、filename、chunk_id、text_preview 和 score，前端可以展示和高亮来源。

### 10. 工程化与可靠性

**Q56：项目如何做配置管理？**  
A：使用 Pydantic Settings 从 `.env` 读取模型、Embedding、Redis、Chroma、Neo4j、chunk、检索和缓存等配置，并做类型校验和默认值管理。

**Q57：为什么用 Docker Compose？**  
A：项目依赖 Neo4j 和 Redis，用 Docker Compose 可以快速启动一致的本地环境，同时持久化 Neo4j 和 Redis 数据。

**Q58：启动时如何恢复系统状态？**  
A：后端启动时会打开 ChromaDB 获取持久化统计，重建前端可见的文档记录，尝试恢复或重建 BM25，并刷新 Neo4j 统计。这样重启后不需要重新上传文档。

**Q59：系统有哪些容错设计？**  
A：Redis 不可用会跳过缓存，Neo4j 构建失败不阻塞主流程，BM25 恢复失败会尝试重建，Planner 解析失败有规则 fallback，Validator 和 Critic 都有最大迭代限制，数据库连接在 finally 中关闭。

**Q60：测试覆盖了哪些内容？**  
A：测试包括 TaskRegistry、Redis 缓存、RAGAS evaluator、RAGAS evaluation agent、QueryDecomposer、GraphRetrieval、测试问题生成、FastAPI app、文档删除和完整工作流等关键路径。

**Q61：你怎么证明这个项目不是只停留在 Demo？**  
A：它有完整 API、前端交互、数据库持久化、文档删除一致性、任务进度、缓存、评估输出和测试用例；而且各模块按业务层和基础设施层拆分，具备持续迭代的工程结构。

### 11. 深挖实现

**Q62：QueryDecomposer 的价值是什么？**  
A：复杂问题往往包含多个意图或多个实体关系，单次检索可能只覆盖一部分。QueryDecomposer 把复杂问题拆成子问题后，每个子问题都能独立规划检索策略，召回更全面。

**Q63：Planner 为什么对每个子问题单独规划？**  
A：同一个复杂问题里不同子问题的信息需求可能不同。比如一个子问题问概念解释，适合 vector；另一个问具体术语，适合 BM25；另一个问关系路径，适合 graph。单独规划比全局固定策略更灵活。

**Q64：为什么要在 Validator 重试时强制使用全部检索器？**  
A：如果第一次规划过窄导致证据不足，重试时继续使用同一策略可能还是漏召回。强制扩展到 vector、keyword、graph 是一种保守但有效的补救策略，可以提高第二轮召回覆盖面。

**Q65：为什么 Writer 后还要 Critic？**  
A：检索到证据不等于答案一定好。Writer 可能遗漏问题、引用不完整或表达不清。Critic 相当于质量门禁，把答案质量检查从人工主观感受变成结构化评分和反馈。

**Q66：Agent 越多越好吗？**  
A：不是。Agent 多会增加调用成本、延迟和调试复杂度。这个项目把 Agent 控制在与 RAG 质量强相关的节点上：拆解、规划、召回、验证、生成和审查。每个 Agent 都有明确输入输出和 fallback。

**Q67：如果线上追求低延迟，你会怎么改？**  
A：可以做分级策略：简单问题跳过 QueryDecomposer、Validator 或 Critic；只在复杂问题上启用完整 Agent 流程；降低 top_k；缓存检索结果；异步预计算图谱和 BM25；对 Critic 做抽样或阈值触发。

**Q68：如果文档很多，当前方案有什么瓶颈？**  
A：瓶颈可能在 Embedding 成本、Chroma 查询规模、BM25 重建、Neo4j 图谱构建和 RAGAS 批量评估。优化方向包括增量 BM25、异步图谱构建、分库分集合、按文档权限过滤、批处理队列和检索结果分页。

**Q69：如果要支持多用户和权限隔离怎么做？**  
A：需要在文档、chunk、向量库 metadata、BM25 索引和 Neo4j 节点边上加入 user_id 或 tenant_id，并在所有检索路径上做过滤。缓存 key 也要包含用户或租户维度，避免跨用户命中。

**Q70：如果答案和引用不一致，可能是什么原因？**  
A：可能是 Writer 没按引用编号生成，citation_ids 解析不稳定，重排后 chunk 顺序变化，或 LLM 引用了上下文外信息。解决方式是收紧 Writer prompt，强制只引用提供的 chunk 编号，并在 Critic 中提高 citation 权重。

**Q71：如果向量检索召回不到正确内容，你怎么调？**  
A：先看 chunk 是否切得太碎或太大，再看 embedding 模型和维度是否正确；然后检查 query 是否需要改写或拆解；再调 top_k 和重排；如果是精确术语问题，增加 BM25 权重；如果是关系问题，引导 Planner 使用 graph。

**Q72：如果 BM25 命中很多噪声怎么办？**  
A：可以做停用词处理、字段权重、最小分数过滤、与向量检索融合后重排，或者只在 Planner 判断存在精确实体、编号、标题时启用 BM25。

**Q73：如果 Neo4j 图谱质量不好怎么办？**  
A：需要优化实体抽取和关系抽取，增加实体规范化、同义词合并、关系类型约束和置信度阈值；也可以把图谱作为辅助召回，而不是唯一证据来源。

**Q74：为什么 RAGAS evaluator 要做兼容层？**  
A：不同 RAGAS 版本指标名称和返回格式可能变化，比如 answer_relevancy/response_relevancy。兼容层把外部库差异封装起来，对项目内部统一返回固定指标名。

**Q75：为什么 RAGAS 中要关闭 Qwen thinking？**  
A：RAGAS 评估会生成多个问题变体或内部调用，某些 Qwen thinking 配置可能和 n 参数冲突。项目里通过 `extra_body: {"enable_thinking": False}` 保证评估调用稳定。

### 12. 项目难点与改进

**Q76：项目最大难点是什么？**  
A：最大难点是把 RAG 从单链路变成可控的多 Agent 状态流。需要定义每个 Agent 的职责、共享状态、失败兜底、条件跳转和 metadata，否则流程会变得不可调试。

**Q77：你遇到过什么问题，怎么解决？**  
A：典型问题是 LLM 输出格式不稳定、图谱或缓存服务不可用、检索证据不足。解决方式分别是 JSON 解析校验和 fallback、best-effort 容错、Validator 触发二次检索并扩大检索器范围。

**Q78：这个项目还有哪些可以优化？**  
A：可以做增量索引更新、权限隔离、多租户、检索融合算法优化、图谱实体消歧、流式输出、队列化任务、在线评估看板，以及基于 RAGAS 结果自动调参。

**Q79：如果让你上线生产，你还会补什么？**  
A：会补鉴权、限流、文件安全扫描、任务队列、日志链路追踪、监控告警、数据库迁移、对象存储、权限过滤、灰度评估和更完整的 e2e 测试。

**Q80：你会如何向面试官总结项目价值？**  
A：这个项目的价值在于把 RAG 做成了一个完整工程系统：前端能用，后端可维护，检索可组合，Agent 流程可观测，答案可溯源，质量可评估。它体现的不只是会调 API，而是能设计和落地一个可迭代的 AI 应用架构。

## 三、面试官追问时的短句模板

- **问到架构：** 我把系统拆成接口层、用例层和基础设施层，Agent 工作流单独放在 workflows 中，这样 HTTP 接口和 RAG 核心逻辑解耦。
- **问到 RAG：** 普通 RAG 的瓶颈是召回和幻觉，我这里用混合检索解决召回，用 Validator 和 Critic 降低幻觉，用 RAGAS 做量化评估。
- **问到 Multi-Agent：** 每个 Agent 都有明确职责和状态输入输出，不是为了堆概念，而是对应 RAG 链路上的真实问题。
- **问到缓存：** 缓存不是只按 query 缓存，而是结合知识库状态 token，避免文档更新后命中旧答案。
- **问到图谱：** 图谱主要解决实体关系和多跳问题，它是增强召回路径，不是替代向量检索。
- **问到评估：** 我用 RAGAS 看 faithfulness、answer relevancy、context precision 和 context recall，能区分是检索问题还是生成问题。
- **问到不足：** 当前更适合单机学习和原型验证，生产化还需要多租户、权限过滤、任务队列、监控和增量索引。

## 四、可以主动讲的技术亮点

1. **动态检索规划**：不是固定走向量检索，而是每个子问题单独选择 vector、keyword、graph 和 top_k 配额。
2. **证据不足自动补救**：Validator 低分时回到 retrieval，并强制扩展为全检索路径。
3. **答案质量闭环**：Critic 对 Writer 答案打分，低于阈值则带反馈再生成。
4. **知识库状态感知缓存**：答案缓存绑定文档状态，兼顾性能和正确性。
5. **RAGAS 量化评估**：用评估指标定位召回不足、噪声过多或生成幻觉。
6. **持久化恢复**：服务重启后从 ChromaDB 恢复文档状态，并恢复或重建 BM25。
