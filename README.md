# 商旅-agent-guide（Python）

企业级智能差旅服务：基于 FastAPI 提供对话、行程规划、差标校验与知识库向量检索，适用于与 OA/费控/供应商系统集成的差旅场景。

## 功能概览

- **对话与工具调用**：LangGraph 工作流编排 + 单个旅行 ReAct Agent，内置「行程草稿」「差标校验」工具。
- **健康检查**：探测 Redis、PostgreSQL、Milvus、关键词索引和模型密钥配置，返回 `ok` / `degraded`。
- **文档入库与检索**：文本嵌入写入 Milvus 和 Redis 关键词索引，支持混合检索。

## 架构说明

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Client/UI  │────▶│  FastAPI         │────▶│ TravelOrchestrator │
└─────────────┘     │  /api/v1/chat    │     │  + LLM + Tools     │
                    │  /health         │     └─────────────┘
                    │  /documents/*    │            │
                    └────────┬─────────┘            ▼
                             │              ┌───────────────┐
                    ┌────────┴────────┐     │ itinerary /  │
                    │ Redis │ PG │ Milvus│     │ policy 领域  │
                    └─────────────────┘     └───────────────┘
```

- **应用层**：`app/main.py` 注册路由与生命周期（连接池、向量库）。
- **编排层**：`app/agent/langgraph_orchestrator.py` 使用 LangGraph 组织一个旅行 ReAct Agent，以及上下文、规划、路由、RAG、规则校验和响应处理节点；`app/agent/orchestrator.py` 保留为 legacy fallback。
- **领域层**：`app/domain/travel/` 行程构建、差标规则与校验。
- **基础设施**：`app/services/`（LLM、嵌入、Milvus）、`app/infrastructure/`（可选扩展）。

## 环境要求

- Python 3.11+
- Docker Compose，或本机 PostgreSQL、Redis、Milvus 2.x
- 宿主机运行时需安装 Node.js 18+，以使用仓库自带的 12306 查询脚本
- 兼容 OpenAI API 的聊天模型密钥；知识库功能还需 Embeddings 密钥

## 安装与运行

在仓库根目录运行。首次部署先复制配置，再把大模型密钥填入 `.env`：

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

**大模型配置位置：根目录 `.env`**。至少填写 `OPENAI_API_KEY`；若使用另一个 Embeddings 服务，还需填写 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL` 和对应的 `EMBEDDING_DIMENSIONS`。使用兼容网关时同步修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。修改后重启服务。未填密钥时服务可启动，但对话及知识库相关请求返回 503，健康状态为 `degraded`。

航班和酒店默认使用明确标记的 Demo 数据；真实数据可设置 `TRAVEL_INVENTORY_PROVIDER=amadeus` 并填写 `AMADEUS_CLIENT_ID`/`AMADEUS_CLIENT_SECRET`，或设为 `flyai` 并填写 `FLYAI_API_KEY`。在 Docker 中使用 FlyAI 时还需设置 `INSTALL_FLYAI_CLI=1` 并重新构建。仓库自带的 12306 脚本默认用于火车票查询，无需额外 API Key，但需要网络连通。真实下单、付款和 OA/费控审批接口尚无供应商配置，当前只生成草稿与审批表。

也可以在宿主机运行 Python，先启动依赖并安装项目包：

```bash
docker compose up -d postgres redis etcd minio milvus-standalone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

首次启动 Milvus 通常需要几十秒；以 `docker compose ps` 中 `milvus-standalone` 变为 `healthy` 为准。修改 `EMBEDDING_DIMENSIONS` 后，已有 Milvus 集合的向量维度不会自动迁移，需使用对应维度的新集合或重建现有知识库。

默认依赖已覆盖 TXT、Markdown、HTML、PDF、DOCX、PPTX、XLSX 解析。扫描版 PDF 的 OCR 还需要本机安装 Tesseract 中文语言包；Docker 镜像已包含。Unstructured 增强解析是可选项；需要时设置 `.env` 中的 `INSTALL_OPTIONAL_DEPS=1` 后重新构建，或在宿主机执行 `pip install -r requirements-optional.txt`。检索先进行关键词＋向量召回和 RRF 融合；配置百炼文本排序 API 后，再调用远端模型重排候选文档。

PDF 入库默认启用条款优先的 Embedding 语义切分（`RAG_SEMANTIC_PDF_ENABLED=true`）：短条款保持完整，长条款按相邻句向量的低相似度断点切分；表格另建完整表和带表头的逐行索引。可用 `RAG_SEMANTIC_MIN_CHARS`、`RAG_SEMANTIC_TARGET_CHARS`、`RAG_SEMANTIC_MAX_CHARS` 调整长度。修改切块配置或解析逻辑后，需删除旧文档并重新入库，评测题集也要按新 chunk ID 重新标注。

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>
- Web 控制台：<http://127.0.0.1:8000/app/>

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务名与文档链接 |
| GET | `/api/v1/health` | 依赖和模型配置状态 |
| POST | `/api/v1/chat` | 对话；`stream: true` 时返回 SSE |
| POST | `/api/v1/documents/ingest` | 文档入库（需 Embeddings、Milvus、Redis） |
| GET | `/api/v1/documents/search` | 混合检索 |
| POST | `/api/v1/mcp/rpc` | MCP JSON-RPC 工具与资源接口 |

### POST `/api/v1/chat`

请求体（节选）：

```json
{
  "messages": [
    { "role": "user", "content": "下周从北京去上海出差一天，帮我估费用并看差标。" }
  ],
  "stream": false,
  "session_id": "optional-session-id"
}
```

- `stream: false`：返回 JSON，结构与 OpenAI Chat Completions 类似（`choices[0].message.content`）。
- 响应同时包含结构化增强字段：`tables`（表格化行程/差标）、`citations`（RAG 引用）、`approval_form`（人工审批单）、`trace`（工作流节点执行轨迹）、`risk_level`（风险等级）。制度问答另含 `rag_evidence`（问题要求、用户事实、制度事实及计算结果）、`rag_stages`（各阶段输出）和 `verification`（逐条结论核验）。多任务响应中，这些问答字段保存在各自的 `task_results` 内。
- `stream: true`：`text/event-stream`，每行 `data: {JSON}`，含 `StreamChunk`（`content` / `done` / `error`）。

### POST `/api/v1/mcp/rpc`

提供轻量 MCP over HTTP JSON-RPC 接口，便于其他 Agent 或工具平台发现并调用本项目能力：

- `initialize`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`

当前暴露的工具：

- `plan_travel_itinerary`
- `check_travel_policy`
- `search_flights`
- `search_hotels`
- `search_trains`
- `recommend_travel_options`

### GET `/api/v1/health`

返回 `status`、`checks`（redis / database / milvus / keyword_index / chat_model_configured / embedding_model_configured；启用重排后另有 rerank_model_configured）、可选 `detail`。密钥检查只确认已配置，不会请求模型接口。

## 配置项

见 `.env.example`。主要变量：`OPENAI_*`、`EMBEDDING_*`、`DATABASE_URL`、`REDIS_URL`、`MILVUS_*`、`LOG_LEVEL`。编排与模型相关阈值（窗口、摘要、熔断）在 `app/config.py` 中定义。

聊天模型和向量模型可分开配置。例如使用 DeepSeek 对话、阿里云百炼文本嵌入：

```bash
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-pro
OPENAI_API_KEY=your_deepseek_key
PLANNER_RESPONSE_FORMAT=json_object

EMBEDDING_BASE_URL=https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=qwen3.7-text-embedding
EMBEDDING_API_KEY=your_bailian_key
EMBEDDING_DIMENSIONS=1536

RAG_RERANKER_ENABLED=true
RAG_RERANKER_URL=https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RAG_RERANKER_MODEL=qwen3.7-text-rerank
RAG_RERANKER_API_KEY=your_bailian_key
```

DeepSeek 的 Chat Completions 接口不支持本项目规划器默认使用的 `json_schema` 响应格式；接入 DeepSeek 时需设置 `PLANNER_RESPONSE_FORMAT=json_object`。重排使用百炼原生 API 的专用地址，不能填 `/compatible-mode/v1`；`RAG_RERANKER_API_KEY` 留空时会回退到 RRF 排序，健康检查显示未配置。修改 `.env` 后重启应用。示例中的 `{WorkspaceId}` 应替换为百炼控制台提供的业务空间 ID。

### LangGraph 工作流模式

默认使用 LangGraph：

```text
context_builder
  ↓
memory_fusion
  ↓
input_guardrail
  ↓
planner
  ↓
intent_router
  ├─ policy_reasoner
  │    ↓
  │  travel_react_agent
  │    ↓
  │  policy_validator
  │    ↓
  │  travel_retry_router ── 可修复且未重试 ──> travel_react_agent
  │    ↓ 通过 / 重试耗尽
  │  approval_processor
  │    ↓
  │  response_reviewer（仅在用户要求复核/挑错时）
  │    ↓
  ├─ rag_responder
  │    ↓
  │  rag_evidence_builder（缺失制度依据时，可补充检索）
  │    ↓
  │  rag_answer_generator
  │    ↓
  │  grounding_verifier
  │    ├─ 首次未通过 ──> rag_self_corrector ──> grounding_verifier
  │    ↓ 通过 / 校正耗尽后保留已支持结论
  └─ general_responder
       ↓
response_finalizer
```

本项目将“Agent”限定为能够自主选择工具、读取工具 observation 并在循环中决定下一步的组件。因此当前在线主链路只有 `travel_react_agent` 属于 Agent；LangGraph 中其他可执行单元统一称为节点。

- **Agent**：`travel_react_agent` 使用 OpenAI function calling 自主选择旅行工具，并在最多 N 轮 ReAct 循环中根据工具结果继续行动或结束。
- **LLM 节点**：`planner`、`policy_reasoner`、`rag_evidence_builder`、`rag_answer_generator`、`rag_self_corrector`、`general_responder`、`response_reviewer` 执行有边界的模型任务。`grounding_verifier` 联合模型语义核验与程序数值校验。
- **规则节点**：`input_guardrail`、`intent_router`、`rag_responder`、`policy_validator`、`travel_retry_router`、`approval_processor` 执行检查、检索、路由或结构化数据处理。
- **上下文与基础设施节点**：`context_builder`、`memory_fusion`、`response_finalizer` 负责会话装配、记忆融合、持久化和响应组装。

可通过环境变量切回旧编排器：

```bash
AGENT_ORCHESTRATOR_BACKEND=legacy
```

候选合规自动重试次数可配置，默认值为 1：

```bash
TRAVEL_VALIDATION_MAX_RETRIES=1
```

### 证据驱动的制度问答

保留关键词和向量召回、RRF、远端 rerank 及 `RAG_FINAL_TOP_K`。取得候选片段后，模型按当前任务整理需要的证据，分开记录用户提供的条件、制度原文和计算。程序检查引文确实来自对应片段、用户条件来自原始问题，使用受限 AST 和 Decimal 计算；公式中的费率和边界必须引用数值事实，不能写成无来源常量。

草稿由带事实 ID 的独立结论组成，程序根据来源生成引用编号。核验模型检查对象、版本、表格列、例外、单位和多来源推导，程序再次检查金额与计算结果。首次核验失败时只校正失败结论，保留原问题及已通过的结论；校正后重新核验。核验不可用或再次失败时仅输出已有核验支持的部分，并说明信息不足，没有已支持结论则拒答。资料中的指令只作为数据处理。

各模型步骤复用 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`，无需新增密钥；服务商需支持 JSON 对象输出。默认每次模型请求最多 120 秒，结构错误和引文错误分别最多修复一次，答案最多校正一次。多任务下每个制度问答任务总超时默认 360 秒，独立查询仍使用 `TASK_TIMEOUT_SECONDS`。整理结果提出缺失制度证据时默认最多补充一次检索；用户未提供的条件、片段明确未载明的信息仍需说明不足。

```bash
RAG_EVIDENCE_TIMEOUT_SECONDS=120
RAG_EVIDENCE_TASK_TIMEOUT_SECONDS=360
RAG_EVIDENCE_MAX_SUPPLEMENTAL_QUERIES=1
RAG_EVIDENCE_CHUNK_MAX_CHARS=8000
```

`verification.passed=true` 表示输出的结论通过当前核验，`question_answered=true` 表示所有所问内容得到回答，两者需分别观察。正确的资料不足说明可通过核验而仍未回答问题；多任务下此类结果为 `needs_review`，不能作为已完成的前置任务。非 RAG 回复的核验为 `status=skipped, passed=null`。模型语义核验仍可能出错，业务正确率需要按评测集人工复核；额外模型步骤也会增加延迟和 token 成本。

## 测试

```bash
pytest
```

（可在 `tests/` 下补充用例。）

真实差旅制度的 RAG 量化试点、证据标注、检索指标与人工答案复核流程见 [评测说明](evals/README.md)。

## 许可证

企业内部使用请以贵司合规要求为准。
