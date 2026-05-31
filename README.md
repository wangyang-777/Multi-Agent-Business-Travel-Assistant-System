# 商旅-agent-guide（Python）

企业级差旅 AI Agent 服务：基于 FastAPI 提供对话、行程规划、差标校验与知识库向量检索，适用于与 OA/费控/供应商系统集成的差旅场景。

## 功能概览

- **对话与工具调用**：LangGraph 多 Agent 编排 + ReAct 风格工具调用，内置「行程草稿」「差标校验」工具。
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
- **编排层**：`app/agent/langgraph_orchestrator.py` 使用 LangGraph 将上下文、意图识别、差旅工具调用、通用回答、复核与最终持久化拆成多个 Agent 节点；`app/agent/orchestrator.py` 保留为 legacy fallback。
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
- 响应同时包含结构化增强字段：`tables`（表格化行程/差标）、`citations`（RAG 引用）、`approval_form`（人工审批单）、`trace`（Agent 执行轨迹）、`risk_level`（风险等级）。
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

见 `.env.example`。主要变量：`OPENAI_*`、`EMBEDDING_*`、`DATABASE_URL`、`REDIS_URL`、`MILVUS_*`、`LOG_LEVEL`。Agent 相关阈值（窗口、摘要、熔断）在 `app/config.py` 中定义。

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

### Agent 编排模式

默认使用 LangGraph：

```text
context_agent
  ↓
guardrail_agent
  ↓
intent_agent
  ├─ travel_react_agent
  │    ↓
  │  approval_agent
  │    ↓
  │  reflection_agent（仅在用户要求复核/挑错时）
  │    ↓
  ├─ rag_agent
  │    ↓
  └─ general_agent
       ↓
finalizer_agent
```

- `context_agent`：加载 Redis 会话、合并历史、摘要长对话、裁剪上下文。
- `guardrail_agent`：执行输入安全与流程边界检查，例如信息不足时阻断预订、拦截明显违规请求。
- `intent_agent`：识别差旅行程、差标、预订、通用问题等意图，并路由到对应节点。
- `travel_react_agent`：使用 OpenAI function calling 调用行程规划与差标校验工具。
- `rag_agent`：对制度、政策、报销、审批等知识类问题检索 Milvus 知识库，并返回引用来源。
- `approval_agent`：根据金额、差标 warning 和工具结果生成 `approval_form`，需要人工审批时标记 `pending_human_approval`。
- `general_agent`：处理不需要差旅工具的普通对话。
- `reflection_agent`：当用户要求“复核/检查/挑错/反思”时，对答案进行质检和修订。
- `finalizer_agent`：保存会话并返回兼容 `/api/v1/chat` 的响应结构。

可通过环境变量切回旧编排器：

```bash
AGENT_ORCHESTRATOR_BACKEND=legacy
```

## 测试

```bash
pytest
```

（可在 `tests/` 下补充用例。）

## 许可证

企业内部使用请以贵司合规要求为准。
