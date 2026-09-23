# 商旅-agent-guide（Python）

企业级智能差旅服务：基于 FastAPI 提供对话、行程规划、差标校验与知识库向量检索，适用于与 OA/费控/供应商系统集成的差旅场景。

## 功能概览

- **对话与工具调用**：LangGraph 工作流编排 + 单个旅行 ReAct Agent，内置「行程草稿」「差标校验」工具。
- **健康检查**：探测 Redis、PostgreSQL、Milvus 可用性，返回 `ok` / `degraded`。
- **文档入库与检索**：文本嵌入（OpenAI Embeddings）写入 Milvus，支持相似度检索。

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
- PostgreSQL（异步 URL）、Redis、Milvus 2.x（可选；未启动时健康检查为 degraded，文档接口可能返回 503）
- 兼容 OpenAI API 的密钥与 `base_url`（含国内兼容网关）

## 安装与运行

```bash
cd project-python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # 编辑 OPENAI_API_KEY 等
```

启动开发服务：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

如果你在宿主机直接运行 `uvicorn`，需要先启动本地依赖：

```bash
docker compose up -d postgres redis etcd minio milvus-standalone
```

`Milvus` 首次启动会依赖 `etcd` 和 `minio` 完成初始化，通常需要几十秒；以 `docker compose ps` 中 `milvus-standalone` 变为 `healthy` 为准，再启动应用更稳妥。

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务名与文档链接 |
| GET | `/api/v1/health` | 依赖健康状态 |
| POST | `/api/v1/chat` | 对话；`stream: true` 时返回 SSE |
| POST | `/api/v1/documents/ingest` | 文档入库（需 Milvus） |
| GET | `/api/v1/documents/search` | 向量检索 |
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
- 响应同时包含结构化增强字段：`tables`（表格化行程/差标）、`citations`（RAG 引用）、`approval_form`（人工审批单）、`trace`（工作流节点执行轨迹）、`risk_level`（风险等级）。
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

### GET `/api/v1/health`

返回 `status`、`checks`（redis / database / milvus）、可选 `detail`。

## 配置项

见 `.env.example`。主要变量：`OPENAI_*`、`EMBEDDING_*`、`DATABASE_URL`、`REDIS_URL`、`MILVUS_*`、`LOG_LEVEL`。编排与模型相关阈值（窗口、摘要、熔断）在 `app/config.py` 中定义。

聊天模型和向量模型可分开配置。例如使用 DeepSeek 聊天、OpenAI embeddings：

```bash
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_API_KEY=your_deepseek_key

EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your_openai_key
EMBEDDING_DIMENSIONS=1536
```

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
  └─ general_responder
       ↓
response_finalizer
```

本项目将“Agent”限定为能够自主选择工具、读取工具 observation 并在循环中决定下一步的组件。因此当前在线主链路只有 `travel_react_agent` 属于 Agent；LangGraph 中其他可执行单元统一称为节点。

- **Agent**：`travel_react_agent` 使用 OpenAI function calling 自主选择旅行工具，并在最多 N 轮 ReAct 循环中根据工具结果继续行动或结束。
- **LLM 节点**：`planner`、`policy_reasoner`、`rag_responder`、`general_responder`、`response_reviewer` 各执行一次有边界的模型任务，不自行调度其他节点。
- **规则节点**：`input_guardrail`、`intent_router`、`policy_validator`、`travel_retry_router`、`approval_processor`、`grounding_verifier` 执行确定性检查、路由或结构化数据处理。
- **上下文与基础设施节点**：`context_builder`、`memory_fusion`、`response_finalizer` 负责会话装配、记忆融合、持久化和响应组装。

可通过环境变量切回旧编排器：

```bash
AGENT_ORCHESTRATOR_BACKEND=legacy
```

候选合规自动重试次数可配置，默认值为 1：

```bash
TRAVEL_VALIDATION_MAX_RETRIES=1
```

## 测试

```bash
pytest
```

（可在 `tests/` 下补充用例。）

## 许可证

企业内部使用请以贵司合规要求为准。
