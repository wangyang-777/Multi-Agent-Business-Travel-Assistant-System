# 商旅 Agent Guide 总体设计

## 1. 文档说明

本文档基于当前仓库代码整理，描述 `travel-agent-guide` 的现状架构、核心流程、数据设计、部署方式、异常降级策略和后续演进方向。文档中的“当前实现”均可在现有代码中找到对应模块；“设计约束”和“演进建议”用于说明系统边界，不代表相关能力已经上线。

| 项目 | 内容 |
| --- | --- |
| 系统名称 | 商旅 Agent Guide |
| 系统类型 | 企业智能差旅服务 |
| 主要语言 | Python 3.11+ |
| Web 框架 | FastAPI |
| 工作流编排 | LangGraph，支持 legacy 编排器回退 |
| 模型协议 | OpenAI-compatible Chat Completions / Embeddings |
| 核心存储 | Redis、Milvus、关键词索引；PostgreSQL 当前仅建立连接并用于健康检查 |
| 对外协议 | REST、SSE、MCP over HTTP JSON-RPC 2.0 |
| 文档版本 | 1.7 |
| 代码基线日期 | 2026-09-28 |
| Git 基线 | 当前工作区（基于 `1dded93`） |

## 2. 建设目标与范围

### 2.1 建设目标

系统面向企业员工、OA/费控系统和其他 Agent 平台，提供以下能力：

1. 以自然语言接收差旅需求，识别行程规划、库存查询、差标、预订和制度问答等意图。
2. 调用航班、酒店和 12306 火车票能力，生成可解释的候选方案；当工具轨迹包含 `mode=travel_recommendation` 的综合推荐结果时派生预订草稿。
3. 通过一次 LLM 调用联合完成意图识别与任务规划，输出符合 JSON Schema 的业务任务列表；校验后按任务数量进入单任务或多任务分支，按依赖调度，再对旅行结果执行制度与审批校验。
4. 为企业制度文档同时建立向量索引和关键词索引，采用双路召回、RRF 融合和检索后重排，基于最终 Top-5 结果生成有引用的回答。
5. 保存短期会话和长期个人偏好，为多轮对话提供上下文。
6. 同时向 Web/API 客户端和 MCP 客户端开放能力。

### 2.2 当前系统边界

- 当前 LangGraph/MCP 在线链路只生成行程方案、候选比选结果和 `booking_draft`，不会自动下单或付款。只要工具轨迹包含 `mode=travel_recommendation` 的结果，就会尝试派生草稿；即使候选查询部分失败，草稿也可能字段不完整。
- `app/core/tools/booking.py` 中存在独立的本地预订领域函数，但尚未接入当前 LangGraph 或 MCP 在线工具链。该函数直接调用时会本地返回 `confirmed` 状态和确认号，因此“不会下单”仅适用于当前在线调用边界。
- 航班和酒店根据配置使用 Demo、Amadeus 或 FlyAI；Amadeus 需要同时配置 client ID 和 secret，否则回退到 Demo。火车票优先使用本地 12306 Skill，其次使用远程 MCP；两者均未配置时返回空结果，不生成演示车票。
- PostgreSQL 当前只创建异步连接池并执行健康探测，业务数据尚未写入 PostgreSQL。
- SSE 接口当前先完成整次工作流调用，再按字符回放结果，并非模型 token 或工作流事件的实时透传。
- 合并意图与任务规划、按任务依赖执行的能力属于 LangGraph 主链路；legacy 编排器仅作为兼容回退，不提供相同的多任务调度保证。
- Redis、Milvus 或关键词索引不可用时应用仍可启动。Redis 不可用会使会话历史、长期记忆、草稿持久化和会话管理接口不可用；任一检索通道不可用或检索异常时，文档接口按依赖状态返回错误，RAG 回退到无可靠引用的 `llm_fallback` 响应。

## 3. 系统上下文

```mermaid
flowchart LR
    Employee[企业员工]
    OA[OA / 费控 / 审批系统]
    AgentPlatform[其他 Agent 或工具平台]

    System[商旅 Agent Guide]

    LLM[OpenAI-compatible<br/>聊天与向量模型]
    FlightHotel[FlyAI / Amadeus<br/>航班与酒店]
    Railway[12306 Skill / MCP<br/>火车票查询]
    Stores[Redis / Milvus / Keyword Index / PostgreSQL]

    Employee -->|Web 控制台 / REST / SSE| System
    OA -->|REST / MCP| System
    AgentPlatform -->|MCP JSON-RPC| System
    System -->|Chat Completions / Embeddings| LLM
    System -->|库存查询| FlightHotel
    System -->|车次与余票查询| Railway
    System -->|会话、知识、健康探测| Stores
```

## 4. 总体架构

### 4.1 分层架构图

```mermaid
flowchart TB
    subgraph Access["接入层"]
        Web[静态 Web 控制台]
        REST[REST / SSE API]
        MCP[MCP JSON-RPC API]
    end

    subgraph Application["应用与编排层"]
        Routes[FastAPI Routes]
        LG[LangGraphTravelOrchestrator]
        Legacy[TravelOrchestrator<br/>legacy fallback]
        Planner[Planner LLM 节点<br/>意图识别与任务规划]
        Intent[规则路由<br/>按任务数量与任务意图分流]
        MultiTask[多任务执行器<br/>依赖调度与结果汇总]
        Guardrail[输入护栏]
        Policy[制度推理与合规校验]
        Approval[审批与 RAG 答案核验]
    end

    subgraph Domain["领域与工具层"]
        TravelDomain[行程与差标领域模型]
        TravelTools[行程规划 / 差标校验]
        Inventory[航班 / 酒店 / 火车查询]
        RAG[RAG 检索与引用生成]
        DocETL[文档解析 / 分块 / Embedding]
    end

    subgraph Infrastructure["基础设施层"]
        LLM[LLMService + CircuitBreaker]
        Redis[(Redis)]
        Milvus[(Milvus)]
        KeywordIndex[(关键词索引)]
        Reranker[检索后 Reranker]
        PG[(PostgreSQL)]
        Providers[FlyAI / Amadeus / 12306]
        Observe[结构化日志 / Trace]
    end

    Web --> REST
    REST --> Routes
    MCP --> Routes
    Routes --> LG
    Routes -.配置或导入失败.-> Legacy
    LG --> Intent
    LG --> Guardrail
    LG --> Planner
    LG --> MultiTask
    LG --> Policy
    LG --> Approval
    LG --> TravelTools
    LG --> Inventory
    LG --> RAG
    Routes --> DocETL
    TravelTools --> TravelDomain
    Inventory --> Providers
    RAG --> LLM
    RAG --> Milvus
    RAG --> KeywordIndex
    RAG --> Reranker
    Policy --> LLM
    Policy --> Milvus
    Policy --> KeywordIndex
    Policy --> Reranker
    DocETL --> LLM
    DocETL --> Milvus
    DocETL --> KeywordIndex
    LG --> LLM
    LG --> Redis
    Routes --> Redis
    Routes -.健康探测.-> PG
    Routes --> Observe
```

### 4.2 模块职责

| 层次 | 主要模块 | 职责 |
| --- | --- | --- |
| 应用入口 | `app/main.py` | 创建 FastAPI 应用，注册路由与静态资源，初始化 Redis、PostgreSQL、Milvus 和编排器 |
| API | `app/api/routes/` | 对话、会话、文档、健康检查和 MCP 接口；完成请求校验和响应 DTO 组装 |
| 主编排 | `app/agent/langgraph_orchestrator.py` | 定义工作流状态、合并意图与规划的 Planner、确定性路由、多任务依赖调度、旅行 ReAct Agent、制度治理、RAG 核验、复核和最终持久化 |
| 兼容编排 | `app/agent/orchestrator.py` | 提供共享工具执行、会话记忆和 legacy ReAct 循环；也是 LangGraph 编排器的父类 |
| 通用智能组件 | `app/core/agent/` | 提供独立的 Planner、ReAct、Reflection 等可复用实现；目录沿用历史命名，当前主链路只有旅行 ReAct 循环按本文口径称为 Agent |
| 通用意图识别 | `app/core/intent/` | 保留规则快车道和可选 LLM 慢车道的可复用组件；LangGraph 主链路由 Planner 统一识别意图，不再调用该规则识别器重新分类 |
| 旅行工具 | `app/core/tools/travel_search.py` | provider 选择、航班/酒店/火车查询、并行综合推荐和结果归一化 |
| 领域模型 | `app/domain/travel/` | 差旅请求、行程、舱位、员工职级、差标规则和审批判断 |
| 文档 ETL | `app/etl/`、`app/services/document_loader.py` | 多格式文档解析、语义分块（overlap 15%）、token 限制、稳定 chunk ID |
| 模型服务 | `app/services/llm.py`、`app/services/embeddings.py` | OpenAI-compatible 聊天与向量调用；聊天调用带熔断器 |
| 向量存储 | `app/services/milvus_store.py` | 集合初始化、COSINE 向量召回、知识写入、查询和删除 |
| RAG 检索与重排 | `app/core/rag/` | 关键词与向量双路召回、RRF 融合（`k=60`）、候选 Top-20、检索后 rerank 和最终 Top-5 输出 |
| 基础设施扩展 | `app/infrastructure/` | LLM 客户端、通用 Redis/PG/Milvus 封装及可观测性组件，部分尚未接入主链路 |
| 控制台 | `app/static/` | 对话、会话历史、知识库维护、RAG 测试和健康状态页面 |

## 5. LangGraph 工作流设计

### 5.1 工作流状态图

```mermaid
flowchart TD
    Start([START]) --> Context[context_builder<br/>合并会话、摘要、裁剪]
    Context --> Memory[memory_fusion<br/>融合长期事实与偏好]
    Memory --> Guardrail{input_guardrail}
    Guardrail -->|阻断| Finalizer[response_finalizer]
    Guardrail -->|通过| Planner[planner<br/>一次 LLM：意图与任务计划<br/>Schema 与业务校验]
    Planner --> Intent{intent_router<br/>读取已校验计划}

    Intent -->|无效计划或缺少关键参数| Clarification[澄清回复]
    Clarification --> Finalizer
    Intent -->|单任务：旅行意图| PolicyReasoner[policy_reasoner<br/>检索并抽取制度约束]
    Intent -->|单任务：知识意图| RAG[rag_responder]
    Intent -->|单任务：其他意图| General[general_responder]
    Intent -->|多任务| MultiTask[multi_task_executor<br/>隔离子任务状态<br/>按依赖有界调度]
    MultiTask --> Aggregate[逐任务完成业务校验与证据核验<br/>按原计划顺序汇总结果]
    Aggregate --> Finalizer

    PolicyReasoner --> Travel[travel_react_agent<br/>执行计划 + 制度约束 + 工具]
    Travel --> PolicyValidator[policy_validator<br/>按推荐结果生成草稿并合规校验]
    PolicyValidator --> Retry{travel_retry_router<br/>校验后重试决策}
    Retry -->|通过| Approval[approval_processor<br/>审批判断]
    Retry -->|可修复且未重试| Travel
    Retry -->|仍失败/需复核| Approval
    Approval --> AfterAnswer{是否要求复核}
    RAG --> AfterAnswer
    General --> AfterAnswer

    AfterAnswer -->|是| Reflection[response_reviewer]
    AfterAnswer -->|否| Verify[grounding_verifier]
    Reflection --> Verify
    Verify -->|通过或非 RAG 回答| Finalizer
    Verify -->|RAG 首次核验失败| Correct[rag_self_corrector<br/>一次有界自校正]
    Correct --> Verify
    Verify -->|校正后仍失败| RAGFallback[RAG 资料不足回退]
    RAGFallback --> Finalizer
    Finalizer --> End([END])
```

### 5.2 状态对象

`TravelGraphState` 是图中节点共享的工作状态，主要字段如下：

| 字段组 | 字段 | 用途 |
| --- | --- | --- |
| 输入 | `messages`、`session_id`、`user_id` | 原始对话与调用方标识 |
| 上下文 | `effective_messages`、`openai_messages` | 合并 Redis 历史、摘要并裁剪后的模型上下文 |
| 记忆 | `memory_context`、`long_term_memories`、`current_facts` | 当前事实、长期偏好和注入模型的优先级规则 |
| 路由 | `intent`、`route` | 主意图和处理分支；实际子任务按各自 `intent` 路由 |
| 规划 | `execution_plan` | 已校验的主意图、业务任务列表、各任务槽位与依赖、缺失参数和澄清问题 |
| 多任务 | `task_results` | 按计划顺序记录每个任务的状态、回答、引用、核验、工具轨迹和治理结果 |
| 结果 | `answer`、`usage`、`response` | 文本答案、token 使用量和最终兼容响应 |
| 工具 | `tool_trace` | 工具名称、参数、原始输出和候选生成轮次 |
| 治理 | `policy_constraints`、`policy_validation`、`approval_form`、`booking_draft`、`risk_level` | 制度约束、合规检查、审批、预订草稿和风险等级 |
| 重试 | `travel_attempt`、`travel_retry_count`、`travel_retry_feedback`、`travel_retry_exhausted` | 当前候选轮次、重试次数、上一轮反馈和重试耗尽标记 |
| RAG | `citations`、`answer_mode`、`verification`、`claim_evidence_map`、`rag_correction_count` | 引用、回答模式、逐条事实与证据对齐结果及自校正次数 |
| 审计 | `trace`、`reflection_notes` | 节点执行轨迹与反思标记 |

### 5.3 组件口径与节点说明

本文不把每个 LangGraph 节点都称为 Agent。判定标准如下：

- **Agent**：能够自主选择工具，消费工具 observation，并在有界循环中决定继续行动还是结束。当前在线主链路只有 `travel_react_agent` 满足该定义。
- **LLM 节点**：由图确定性调度，完成一次受限模型任务，不自主调度其他节点，包括 `planner`、`policy_reasoner`、`rag_responder`、`rag_self_corrector`、`general_responder` 和 `response_reviewer`。
- **规则节点**：不依赖模型自主决策，执行输入检查、分类路由、合规校验、重试路由、审批表生成或引用核验。
- **基础设施节点**：负责上下文、记忆、持久化和响应组装。

| 节点 | 类型 | 输入重点 | 核心处理 | 输出/去向 |
| --- | --- | --- | --- | --- |
| `context_builder` | 基础设施节点 | 原始消息、session | 加载 Redis 会话；去重合并；长会话摘要；窗口裁剪 | 模型消息上下文 |
| `memory_fusion` | 基础设施节点 | 当前用户文本、user/session | 抽取职级、常驻地、偏好；合并长期记忆；注入优先级规则 | 护栏 |
| `input_guardrail` | 规则节点 | 用户文本 | 阻止绕过审批、伪造发票、提示注入；检查预订所需信息 | Planner 或直接结束 |
| `planner` | LLM 节点 | 用户文本、会话与记忆上下文 | 一次模型调用同时识别意图和拆分业务任务，输出 Schema 约束 JSON；本地校验任务枚举、参数、唯一 ID 和无环依赖 | 已校验计划或澄清结果 |
| `intent_router` | 规则节点 | 已校验 `execution_plan` | 根据 `len(tasks)` 判断单/多任务；单任务仅按任务 `intent` 路由；无效计划和缺参请求进入澄清 | policy reasoner、RAG、general、多任务执行器或最终处理 |
| `multi_task_executor` | 规则节点 | 任务列表与依赖 | 创建隔离子任务状态；有界并行执行就绪的只读任务；依赖任务等待前置结果；复用既有业务与核验节点，按原计划顺序汇总 | 统一最终处理 |
| `policy_reasoner` | LLM 节点 | 用户文本、混合检索、记忆 | 通过关键词与向量双路召回制度资料，RRF（`k=60`）融合并对候选 Top-20 进行 rerank，取最终 Top-5 后由单次 LLM 调用与正则启发式抽取结构化约束 | 旅行 ReAct Agent |
| `travel_react_agent` | Agent | 上下文、执行计划、制度约束、工具定义 | 将计划与约束注入模型；自主 function calling；追加 observation；最多 N 轮 | 合规校验 |
| `policy_validator` | 规则节点 | 制度约束、当前轮工具轨迹 | 仅在综合推荐结果存在时生成草稿；随后校验酒店限额、舱位、火车席别、审批阈值和提前预订天数 | 重试决策 |
| `travel_retry_router` | 规则节点 | 当前轮候选、合规结果、重试计数 | 在校验之后识别空候选、工具错误和可修复差标；最多回到 ReAct 重试一次，耗尽后提升风险并转人工审核 | 旅行 ReAct Agent 或审批节点 |
| `rag_responder` | LLM 节点 | 问题、混合检索与重排结果 | 关键词与向量各召回 Top-20；按 RRF（`k=60`）融合保留候选 Top-20；执行检索后 rerank，最终取 Top-5；使用受限提示词和低温参数生成带逐条引用的回答草稿 | 核验或复核 |
| `general_responder` | LLM 节点 | 模型上下文 | 不带工具的单次通用回答 | 核验或复核 |
| `approval_processor` | 规则节点 | 工具轨迹、预订草稿 | 基于工具结果和草稿生成审批表，补充人工确认与审批提示 | 核验或复核 |
| `grounding_verifier` | 规则节点 | 回答、最终 Top-5、回答模式、校正次数 | 将回答拆为原子事实，逐条与检索证据对齐并标记 `supported`、`partial`、`unsupported` 或 `conflict`；仅全部可核验事实受支持时通过 | 最终处理、自校正或资料不足回退 |
| `rag_self_corrector` | LLM 节点 | 原回答、逐条核验结果、最终 Top-5 | 最多执行一次有界自校正；删除或改写无依据陈述、补齐引用，但不得引入 Top-5 以外的新事实 | 重新进入 grounding verifier |
| `response_reviewer` | LLM 节点 | 初稿 | 按日期、城市、金额、差标、审批风险执行一次质检和修订 | 最终处理 |
| `response_finalizer` | 基础设施节点 | 全部状态 | 强制企业输出契约；统一保存会话和草稿；组装单任务或多任务元数据；子任务不重复调用 | 最终响应 |

### 5.4 合并意图识别与任务规划

意图枚举包括 `search_flight`、`search_hotel`、`search_train`、`trip_planning`、`application`、`policy`、`booking`、`info_query`、`rag` 和 `general`。

所有通过输入护栏的请求都进入 `planner`。该节点使用一次 LLM 调用同时完成意图识别、任务拆分、参数抽取和依赖规划，模型返回符合 JSON Schema 的 JSON 数据。这里的“一次”只限定意图与规划阶段；会话摘要、RAG 回答、制度抽取及 ReAct 仍可能各自调用模型。

模型输出契约如下：

| 字段 | 约束与含义 |
| --- | --- |
| `primary_intent` | 必须与至少一个子任务意图一致，描述整个请求，供展示和审计；不覆盖子任务路由 |
| `tasks` | 1～8 个业务任务；任务数量由程序计算 |
| `tasks[].id` | 任务唯一标识，用于依赖和结果关联 |
| `tasks[].intent` | 当前任务的意图枚举，决定其业务分支 |
| `tasks[].request` | 完整、可独立理解的任务描述，保留用户约束 |
| `tasks[].slots` | 符合槽位 Schema 的结构化参数；严格输出要求列出所有槽位键，不适用或未知时为 `null`，不得编造 |
| `tasks[].depends_on` | 前置任务 ID 列表；引用必须存在，不得依赖自身或形成环 |
| `tasks[].missing_slots` | 缺失的必要参数名称 |
| `clarification_question` | 需要澄清的问题，无需澄清时为 `null` |

以“查询 2026 年 10 月 1 日北京到上海的航班，并说明公司报销流程”为例，任务结构如下。此处为便于阅读省略了值为 `null` 的槽位；实际严格结构化输出需包含 `TaskSlots` 定义的全部键。

```json
{
  "primary_intent": "search_flight",
  "tasks": [
    {
      "id": "task_1",
      "intent": "search_flight",
      "request": "查询2026年10月1日北京到上海的航班",
      "slots": {"origin": "北京", "destination": "上海", "depart_date": "2026-10-01"},
      "depends_on": [],
      "missing_slots": []
    },
    {
      "id": "task_2",
      "intent": "policy",
      "request": "查询公司差旅报销流程",
      "slots": {},
      "depends_on": [],
      "missing_slots": []
    }
  ],
  "clarification_question": null
}
```

默认使用模型接口的 `json_schema` 严格结构化输出。供应商不支持该能力时，可显式配置为 `json_object`；该模式仍执行相同的本地 Schema 与业务校验，不以“能解析 JSON”代替合法计划，也不在失败后自动追加第二次规划模型调用。

Schema 定义位于 `app/domain/task_plan.py`，与在线工具参数对齐：航班使用 `origin`、`destination`、`depart_date`，酒店使用 `city`、`check_in`、`check_out`，火车使用 `origin_station`、`dest_station`、`depart_date`，综合行程使用 `origin_city`、`destination_city`、`departure_date` 等。日期必须合法，金额和人数有类型及范围限制。

校验包括字段类型与额外字段、意图白名单、任务数量、任务 ID 长度、非空文本、槽位类型、唯一任务 ID、依赖引用和无环性。缺少关键参数、模型调用失败、空任务列表或非法计划均返回澄清，停止本次业务工具执行；不按关键词猜测计划，也不将复合请求静默缩减为一个任务。

### 5.5 单任务路由与多任务调度

`intent_router` 根据已校验的 `tasks` 数量确定分支，不再用动作词、业务名词或分类置信度推断自然语言是否包含多个任务，也不另调意图分类 LLM。

- 只有一个任务：旅行意图进入 `policy_reasoner → travel_react_agent`；知识类意图进入 `rag_responder`；其余意图进入 `general_responder`。
- 多于一个任务：进入 `multi_task_executor`，每个子任务按自己的 `intent` 复用相同业务分支。

任务粒度是能够独立交付结果的业务请求。“查航班并说明报销流程”是两个任务；“规划符合差标的出差方案”可以是一个 `trip_planning` 任务，内部包含制度检索和多个工具步骤。工具调用次数不能作为任务数量。

多任务执行器按依赖关系选取就绪任务。没有相互依赖的只读任务可以并行，默认并发上限为 3；依赖任务必须等前置任务成功后执行，并接收前置任务结果。涉及草稿或申请的业务任务顺序执行。默认规划超时为 30 秒，单个子任务超时为 120 秒。每个任务有独立失败状态；前置任务失败时，其依赖任务标记为阻塞，不将未完成任务报告为成功，也不丢弃其他已完成结果。

每个子任务使用隔离状态，保存自身回答、工具轨迹、引用、合规结果及事实核验。子任务执行不重复调用 Planner、上下文初始化、记忆写入或最终持久化。RAG 回答在子任务范围完成事实核验及有界自校正；汇总节点按原计划顺序组装已核验回答和状态，不通过额外生成改写掩盖失败或改变引用。顶层只进行一次会话及草稿持久化，并保留 `task_results` 以便 API 和控制台逐项展示。

任务状态为 `completed`、`needs_review`、`failed` 或 `blocked`，分别表示完成、需要人工复核、执行失败和前置条件未满足。多任务响应的顶层 `booking_draft`、`approval_form` 保持为空，各任务的草稿及审批结果保存在对应 `task_results` 条目中，避免不同任务的结果互相覆盖。

## 6. 核心业务时序

### 6.1 对话、工具调用与审批时序

```mermaid
sequenceDiagram
    autonumber
    actor User as 用户/调用方
    participant API as FastAPI /chat
    participant Graph as LangGraph 编排器
    participant Redis as Redis
    participant LLM as Chat Model
    participant Executor as 业务分支/子任务执行器

    User->>API: POST /api/v1/chat
    API->>Graph: run_completion(messages, session_id, user_id)
    Graph->>Redis: 读取 chat:session:{session_id}
    Redis-->>Graph: 历史消息
    Graph->>Redis: 读取/更新 memory:long:{owner}
    Graph->>Graph: input_guardrail 合规检查
    Graph->>LLM: planner 联合识别意图与规划（一次调用，JSON Schema）
    LLM-->>Graph: primary_intent / tasks / clarification_question
    Graph->>Graph: Schema、槽位和依赖图校验
    alt 调用失败、非法计划或缺少关键参数
        Graph->>Graph: 构造澄清回复，不调用业务工具
    else 已校验的单任务
        Graph->>Executor: 按 tasks[0].intent 执行对应业务分支
        Executor->>Executor: 旅行合规 / RAG 核验 / 可选复核
        Executor-->>Graph: answer、引用、治理信息
    else 已校验的多任务
        loop 依赖图仍有可执行任务
            Graph->>Executor: 就绪任务；独立只读任务有界并行
            Executor->>Executor: 隔离上下文，复用业务分支及核验节点
            Executor-->>Graph: 每任务状态、结果、引用、核验、工具轨迹
            Graph->>Graph: 更新依赖状态，记录失败或阻塞
        end
        Graph->>Graph: 按原计划顺序汇总 task_results 和回答
    end
    Graph->>Redis: 统一保存一次会话、各任务草稿及 TTL
    Graph-->>API: Chat completion + 结构化元数据
    API->>API: 转换计划、任务结果、制度、合规、库存和审批表格
    API-->>User: JSON 或 SSE 字符流
```

### 6.2 旅行计划与制度治理时序

```mermaid
sequenceDiagram
    autonumber
    participant Planner as planner
    participant Intent as intent_router
    participant Policy as policy_reasoner
    participant Milvus as Milvus
    participant Keyword as Keyword Index
    participant RRF as RRF Fusion
    participant Reranker as Reranker
    participant React as travel_react_agent
    participant Validator as policy_validator
    participant Retry as travel_retry_router
    participant Approval as approval_processor

    Planner->>Planner: 一次 LLM 生成并校验 execution_plan
    Planner->>Intent: primary_intent、任务意图、槽位和依赖
    Intent->>Policy: 已校验的单任务，或多任务调度器选出的旅行子任务
    Policy->>Milvus: 向量召回制度与差标资料 Top-20
    Policy->>Keyword: 关键词召回制度与差标资料 Top-20
    Milvus-->>Policy: vector candidates + ranks
    Keyword-->>Policy: keyword candidates + ranks
    Policy->>RRF: RRF 融合（k=60），保留候选 Top-20
    RRF-->>Policy: fused candidates
    Policy->>Reranker: 对候选 Top-20 执行 rerank
    Reranker-->>Policy: 最终 Top-5 citations
    alt 检索成功且存在资料
        Policy->>Policy: LLM 抽取 + 正则启发式合并约束
    else 不可用或检索失败
        Policy->>Policy: 生成空约束并标记来源
    end
    Policy->>React: execution_plan + policy_constraints
    React->>React: 调用库存/差标工具并形成回答
    React->>Validator: tool_trace
    opt tool_trace 包含 travel_recommendation
        Validator->>Validator: 创建 booking_draft
    end
    Validator->>Validator: 基于现有约束和草稿逐项校验制度
    Validator->>Retry: policy_validation + booking_draft
    alt 校验通过
        Retry->>Approval: 继续审批判断
    else 候选为空、工具失败或差标可通过换候选修复，且 retry_count < 1
        Retry->>React: 上一轮违规/缺失原因
        React->>React: 重新调用查询/推荐工具形成第二轮候选
        React->>Validator: attempt=2 的 tool_trace
        Validator->>Retry: 第二轮 policy_validation
        alt 第二轮通过
            Retry->>Approval: 使用第二轮候选
        else 第二轮仍不符合或无合适候选
            Retry->>Retry: 标记 retry_exhausted，risk_level=high
            Retry->>Approval: 强制 pending_human_approval
        end
    else 制度缺失、审批阈值或日期等不可通过换候选修复
        Retry->>Approval: 直接进入人工复核
    end
    opt tool_trace 非空
        Approval->>Approval: 根据风险生成 approval_form
    end
```

### 6.3 RAG 文档入库时序

```mermaid
sequenceDiagram
    autonumber
    actor Admin as 知识库管理员
    participant API as Documents API
    participant Loader as Document Loader
    participant ETL as Chunk Pipeline
    participant Embed as Embedding Service
    participant Milvus as Milvus
    participant Keyword as Keyword Index

    Admin->>API: 上传文件或提交长文本
    opt 文件上传
        API->>Loader: 解析支持的 txt/md/html/htm/pdf/docx/pptx/xlsx 文件
        alt PDF
            Loader->>Loader: 页面级检测文本、表格和图片区域
            Loader->>Loader: 先解析表格结构和图片内容
            Loader->>Loader: 扫描页/图片文字区域按需 OCR
            Loader->>Loader: 剩余正文按段落恢复阅读顺序
        end
        Loader-->>API: 文本 + 结构化块 + 来源元数据
    end
    API->>ETL: 清洗并按标题/段落/句子等语义单元分块
    ETL->>ETL: 相邻 chunk 保留前一 chunk 尾部 15% overlap
    ETL->>ETL: 仅在超出 embedding 上限时执行 token 安全拆分
    ETL-->>API: chunks + 稳定 chunk IDs
    API->>Embed: 批量生成 embeddings
    Embed-->>API: vectors
    API->>Milvus: insert_vectors
    Milvus->>Milvus: 写入并 flush
    Milvus-->>API: 完成
    API->>Keyword: upsert chunks、文本和元数据
    Keyword-->>API: 关键词索引完成
    API-->>Admin: doc_id、chunk_count、vector_dim
```

### 6.4 RAG 问答与降级时序

```mermaid
sequenceDiagram
    autonumber
    participant Graph as rag_responder
    participant Embed as Embedding Service
    participant Milvus as Milvus
    participant Keyword as Keyword Index
    participant RRF as RRF Fusion
    participant Reranker as Reranker
    participant LLM as Chat Model

    alt 双路检索服务可用
        Graph->>Embed: embed_text(question)
        Embed-->>Graph: query vector
        Graph->>Milvus: 向量召回 Top-20
        Graph->>Keyword: 关键词召回 Top-20
        Milvus-->>Graph: vector candidates + ranks
        Keyword-->>Graph: keyword candidates + ranks
        Graph->>RRF: 合并双路候选，计算 RRF（k=60）
        RRF-->>Graph: 融合候选 Top-20
        Graph->>Reranker: 对融合候选 Top-20 执行 rerank
        Reranker-->>Graph: 最终 Top-5 chunks + rerank scores
        Graph->>Graph: 对最终 Top-5 执行引用可靠性检查
        alt 引用可靠
            Graph->>LLM: 受限提示词 + 最终 Top-5 + 问题，temperature=0.1
            LLM-->>Graph: 带逐条引用的 grounded answer 草稿
            Graph->>Graph: 将回答拆为原子事实并与 Top-5 证据对齐
            alt 所有可核验事实均有直接依据
                Graph->>Graph: verification=passed
            else 存在 partial/unsupported/conflict
                Graph->>LLM: 原回答 + 失败事实 + 对应证据，执行一次自校正
                LLM-->>Graph: 删除/改写无依据内容后的校正回答
                Graph->>Graph: 再次执行事实-证据对齐
                alt 二次核验通过
                    Graph->>Graph: verification=passed_after_correction
                else 二次核验仍失败
                    Graph->>Graph: 回退为资料不足回答，不输出无依据制度结论
                end
            end
        else 无可靠引用
            Graph->>LLM: 标注资料不足的 fallback prompt
            LLM-->>Graph: llm_fallback answer
        end
    else 任一路检索不可用或检索异常
        Graph->>LLM: 标注知识库检索不可用的 fallback prompt
        LLM-->>Graph: llm_fallback answer
        Graph->>Graph: 仍按路由进入 grounding_verifier 或 response_reviewer
    end
```

### 6.5 MCP 工具调用时序

```mermaid
sequenceDiagram
    autonumber
    participant Client as MCP Client
    participant API as POST /api/v1/mcp/rpc
    participant Dispatcher as MCP Dispatcher
    participant Orch as Orchestrator
    participant Tool as Travel Tool

    Client->>API: initialize
    API->>Dispatcher: dispatch
    Dispatcher-->>Client: protocolVersion + capabilities
    Client->>API: tools/list
    Dispatcher-->>Client: 6 个工具及 inputSchema
    Client->>API: tools/call(name, arguments)
    Dispatcher->>Orch: _execute_tool(name, JSON arguments)
    Orch->>Tool: 执行领域或库存能力
    Tool-->>Orch: 文本/JSON
    Orch-->>Dispatcher: output
    Dispatcher-->>Client: MCP text content
```

## 7. 工具与外部供应商设计

### 7.1 在线工具清单

| 工具 | 作用 | 主要依赖 |
| --- | --- | --- |
| `plan_travel_itinerary` | 基于结构化需求生成行程草稿和费用预估 | 本地领域模型 |
| `check_travel_policy` | 校验舱位、预算、提前预订和审批条件 | 本地差标规则 |
| `search_flights` | 查询航班候选 | demo / FlyAI / Amadeus |
| `search_hotels` | 查询酒店候选并支持预算、POI 过滤 | demo / FlyAI / Amadeus |
| `search_trains` | 查询高铁/火车候选和余票 | 本地 12306 Skill 或远程 MCP |
| `recommend_travel_options` | 并行查询航班、酒店和火车，排序后形成组合推荐 | 上述 provider |

`app/core/tools/booking.py` 虽然定义了 `create_booking`，但不在上述在线工具白名单和执行分发中；当前在线流程只使用推荐结果派生 `BookingDraft`。

### 7.2 Provider 选择

```mermaid
flowchart TD
    Config[TRAVEL_INVENTORY_PROVIDER] --> Choice{配置值}
    Choice -->|flyai| FlyAI[FlyAI CLI]
    Choice -->|amadeus 且凭据完整| Amadeus[Amadeus HTTP API]
    Choice -->|其他或凭据不足| Demo[Demo Provider]

    RailConfig{火车票配置}
    RailConfig -->|RAILWAY_12306_SKILL_DIR| Skill[本地 Node 12306 Skill]
    RailConfig -->|否则 RAILWAY_MCP_URL| RailMCP[远程 12306 MCP]
    RailConfig -->|均未配置| Unavailable[返回 provider_not_configured]
```

外部查询失败时，系统返回空候选、错误原因和免责声明，不会静默替换为伪造的实时数据。综合推荐通过 `asyncio.gather` 并行查询各类库存，再选取排序后的首选候选；只要返回 `travel_recommendation` 载荷，就会尝试生成草稿，候选为空时草稿中的推荐项可能为空。

## 8. 数据与存储设计

### 8.1 数据关系

```mermaid
erDiagram
    CHAT_SESSION ||--o{ CHAT_MESSAGE : contains
    USER_OR_SESSION ||--o{ LONG_TERM_MEMORY : owns
    CHAT_SESSION ||--o| BOOKING_DRAFT : latest
    KNOWLEDGE_DOCUMENT ||--|{ KNOWLEDGE_CHUNK : split_into

    CHAT_SESSION {
        string session_id PK
        json messages
        int ttl_seconds
    }
    CHAT_MESSAGE {
        string role
        string content
        string name
        string tool_call_id
    }
    USER_OR_SESSION {
        string owner_id PK
    }
    LONG_TERM_MEMORY {
        json facts
        int ttl_seconds
    }
    BOOKING_DRAFT {
        string draft_id PK
        string status
        json recommendations
        decimal estimated_total_cny
        boolean approval_required
        datetime expires_at
    }
    KNOWLEDGE_DOCUMENT {
        string parent_doc_id PK
        string title
        string doc_type
    }
    KNOWLEDGE_CHUNK {
        string id PK
        string title
        string doc_type
        string content
        vector embedding
    }
```

> 该图描述的是业务逻辑对象及其关系，并非当前 PostgreSQL 的物理表结构。当前会话、记忆和草稿以 Redis 键保存；知识内容以带重复元数据的 Milvus chunk 保存，并同步维护关键词索引，尚未持久化独立的 `KNOWLEDGE_DOCUMENT` 业务实体。

### 8.2 文档类型与解析能力

当前文档上传入口按文件扩展名识别并处理以下类型。扩展名会先统一转为小写；解析成功后统一转换为纯文本，再进入语义分块、embedding 和双路索引入库。

| 文件类型 | 当前解析方式 | 处理范围与限制 |
| --- | --- | --- |
| `.txt` | 内置文本解析 | 支持 UTF-8、UTF-8 BOM 和 GB18030 等常见编码；按纯文本处理 |
| `.md` | 优先 Unstructured，失败时使用内置文本解析 | 支持 Markdown 文本；无 Unstructured 时仍可按纯文本回退 |
| `.html` / `.htm` | Unstructured | 提取网页正文、标题等元素；需要本地 Unstructured 解析依赖 |
| `.pdf` | 优先使用 PyMuPDF 执行版面文本、表格和图片信息解析，低文本密度页面按需调用 Tesseract OCR；PyMuPDF 不可用时回退 `pypdf` | 表格转换为 Markdown；保留页码、表格/图片数量、OCR 标记、置信度和 `needs_review` 信息。复杂跨页表格和高密度图文页面仍需持续评测 |
| `.docx` | 优先 Unstructured，失败时使用 `python-docx` | 支持段落和表格文本；无 Unstructured 时保留内置回退路径 |
| `.pptx` | Unstructured | 提取演示文稿中的文本元素；需要本地 Unstructured 解析依赖 |
| `.xlsx` | Unstructured | 提取工作表/表格中的文本元素；需要本地 Unstructured 解析依赖 |

知识库的 `doc_type` 是业务分类而不是文件扩展名，当前允许的取值为 `policy`（制度）、`sop`（流程）、`city_guide`（城市指南）和 `other`（其他）。文件格式支持不等于所有内容都能成功提取；空文档、受密码保护的文件、损坏文件或无可提取文本的文件会在解析阶段返回错误。

独立源代码文件扩展名当前不在上传白名单中；下述“代码”分块约束当前适用于 Markdown 等受支持文档中的代码块。后续若开放 `.py`、`.java`、`.js`、`.ts` 等源代码文件上传，必须复用同一函数/类级分块约束，不得退化为固定长度切分。FAQ 是内容结构而不是文件扩展名，可来自 TXT、Markdown、HTML、DOCX、PDF 或其他受支持载体。

### 8.3 Redis 键设计

| 键模式 | 内容 | TTL |
| --- | --- | --- |
| `chat:session:{session_id}` | 最近最多 `memory_max_messages` 条消息 | 默认 86,400 秒 |
| `memory:long:{user_id或session_id}` | 去重后的长期偏好/事实，最多 20 条 | 会话 TTL 的 30 倍 |
| `booking:draft:{draft_id}` | 完整预订草稿 JSON | 默认 86,400 秒 |
| `booking:session:{session_id}:latest` | 会话最近草稿 ID | 默认 86,400 秒 |

预订草稿对象自身的 `expires_at` 为创建后 30 分钟，它表示业务有效期；Redis 键 TTL 表示技术保留期，两者含义不同。

### 8.4 Milvus 集合设计

集合名称为 `travel_knowledge`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | VARCHAR(64)，主键 | 单文档 ID 或 `chk_{parent_doc_id}_{hash}` |
| `title` | VARCHAR(512) | 文档标题 |
| `doc_type` | VARCHAR(32) | `policy`、`sop`、`city_guide` 或 `other` |
| `content` | VARCHAR(65535) | 文档或 chunk 正文 |
| `embedding` | FLOAT_VECTOR(1536) | 文本向量 |

索引使用 `IVF_FLAT`，距离度量为 `COSINE`，默认 `nlist=128`，检索时 `nprobe=16`。

### 8.5 文档分块策略

文档分块不采用固定字符长度作为主切分规则，而采用语义切分：优先按标题、章节、段落、列表项、表格块和句子边界组织 chunk，尽量保持一个 chunk 内主题完整、上下文连续。字符数或 token 数只作为安全上限，不作为常规切分依据。

不同内容类型必须遵守以下分块约束：

| 内容类型 | 强制处理与分块规则 | 必须保留的元数据 |
| --- | --- | --- |
| Markdown | 按标题层级组织和切分内容；每个 chunk 必须携带从一级标题到当前标题的完整标题路径。标题下的列表、引用、表格和代码块作为语义单元处理，不能仅按字符数截断 | `heading_path`、标题级别、源文件、块类型 |
| PDF | 先执行页面版面分析，识别并解析表格和图片区域，再对剩余正文按阅读顺序和段落切分。表格转换为保留表头和行列关系的 Markdown/结构化文本；大表按完整行分块并在每个 chunk 重复表头。图片需保留图注和邻近正文，包含文字时执行 OCR；扫描页或无法直接提取文本的页面必须执行 OCR | 页码、表格/图片编号、区域坐标、提取方式、OCR 标记与置信度、标题/图注 |
| 代码 | 使用语言解析器或 AST 按函数、方法、类切分；保留签名、装饰器/注解、文档字符串及完整代码块。模块级导入、常量和上下文按依赖关系附加，不得从任意行中间截断函数或类 | 语言、文件路径、模块、类名、函数名、起止行、符号路径 |
| FAQ | 每个“问题 + 答案”作为一个不可分割的 chunk；不得把问题与答案拆到不同 chunk，也不得把相邻问答错误合并。分类、同义问法和标签作为元数据保存 | FAQ ID、问题、分类、标签、来源 |

- 相邻 chunk 的 overlap 比例固定为前一 chunk 内容的 15%，优先完整保留句子或段落尾部；如果 15% 的边界落在句子中间，应向相邻语义边界对齐。
- 15% overlap 只作用于允许重叠的自然语言段落边界；不得通过 overlap 拆散或部分复制 FAQ 问答对、完整函数/类、表格行或图片语义单元。Markdown chunk 的重叠内容仍须携带正确的标题路径。
- 只有当语义单元超过 embedding 模型允许的 token 上限时，才进行递归安全拆分；普通正文按段落/句子边界拆分，大表格按完整行拆分并重复表头。超长代码块按嵌套符号边界处理，超长 FAQ 问答对进入大单元处理或人工复核；禁止对代码和 FAQ 直接做无语义 token 截断。
- 每个 chunk 使用稳定 `chunk_id`，相同 chunk 文本同步写入 Milvus 与关键词索引；标题、业务类型写入两侧，标题路径、页码、OCR 和块类型等详细元数据保存在关键词索引中，并在融合时按 `chunk_id` 回填，保证去重、RRF 融合和引用追踪一致。
- 分块结果需要保留标题路径、页码、表格/图片来源、代码符号和 FAQ 标识等可用元数据；这些元数据随最终 Top-5 引用返回，但不参与正文内容的重复拼接。

PDF 的 OCR 采用按需触发，而不是对所有页面强制执行。页面无可提取文本、文本密度低于质量阈值、页面主要由图片构成，或图片/表格区域包含无法直接提取的文字时触发 OCR。OCR 只负责识别文字，表格仍必须经过行列结构恢复；OCR 或表格解析置信度低于质量阈值时，该 chunk 标记为 `needs_review`，不得直接作为高可信制度依据。

### 8.6 RAG 检索与排序策略

RAG 对用户问题采用关键词和向量两路独立召回，再进行融合和检索后重排。关键词索引当前以 Redis Hash 持久化完整 chunk，并采用适配中英文混合文本的 BM25 排序；应用启动时会使用已有 Milvus chunk 补齐关键词索引。入库时每个 chunk 以同一个稳定 `chunk_id` 同步写入 Milvus 和关键词索引，保证两路结果可以去重、合并并追踪到同一引用。两路召回使用相同的候选规模，融合后仍限制候选规模，避免将过多片段送入重排模型。

| 阶段 | 策略 | 参数/输出 |
| --- | --- | --- |
| 关键词召回 | 基于 Redis chunk 语料的 BM25 词法检索 | 返回关键词候选 Top-20，并记录关键词侧 rank；当前实现按请求计算 BM25，规模扩大后应迁移到专用倒排引擎 |
| 向量召回 | Query embedding + Milvus COSINE 检索 | 返回向量候选 Top-20，并记录向量侧 rank |
| 双路融合 | Reciprocal Rank Fusion（RRF） | `RRF(d) = Σ 1 / (60 + rank_i(d))`；`k=60`，未出现在某一路的文档该路贡献为 0 |
| 候选截断 | 按 RRF 分数降序 | 保留融合候选 Top-20 |
| 检索后重排 | 对融合候选执行 Cross-Encoder rerank（query-document pair） | 输入 Top-20 候选，按 rerank 分数重新排序 |
| 最终返回 | 取重排结果前缀 | 最终只向回答生成和引用核验返回 Top-5 |

排序链路固定为：

```text
关键词 Top-20 + 向量 Top-20
    -> RRF 融合（k=60）
    -> 融合候选 Top-20
    -> rerank
    -> 最终 Top-5
```

RRF 只使用各召回通道的名次，不直接比较关键词分数和向量相似度的量纲；rerank 分数只用于最终 Top-5 排序，不替代 RRF 的候选生成职责。最终 Top-5 的原始召回分数、RRF 分数和 rerank 分数应保留在引用及检索 trace 的元数据中，便于评测和问题定位。

Cross-Encoder 模型采用延迟加载。模型依赖或模型文件暂时不可用时，系统记录 `rerank_status=rrf_fallback` 并按 RRF 顺序返回 Top-5，不会绕过双路召回或扩大候选范围；生产部署应预下载并固定模型版本，避免首次请求触发模型下载。

### 8.7 RAG 生成防幻觉与事实对齐

RAG 回答必须经过“受限生成、事实对齐、一次自校正、失败回退”四层控制。检索命中本身不等于回答可信；只有最终 Top-5 能直接支持回答中的全部可核验事实时，回答才能标记为 `rag_grounded`。

| 控制项 | 设计约束 |
| --- | --- |
| 提示词限制 | System prompt 明确要求只能使用最终 Top-5 中的内容回答；禁止使用模型记忆补充公司制度、金额、期限、职级、适用范围或审批条件；资料未说明时必须明确回答“检索资料未提供该信息”；不得引用不存在的编号，也不得将通用建议表述为公司规定 |
| 低温生成 | RAG grounded answer 和自校正默认使用 `temperature=0.1`；不得通过提高温度获得更多制度性内容。需要表达多样性的非事实性文案可单独处理，但不能改变事实结论 |
| 输出自校正 | 初稿核验失败时最多执行一次 `rag_self_corrector`。校正器只能删除、收缩或依据 Top-5 改写失败陈述并修复引用，不能引入新证据或新的事实断言；校正后必须再次核验 |
| 检索-生成对齐 | 将回答拆分为最小可核验的原子事实，为每条事实绑定一个或多个 `chunk_id` 和原文证据片段，并记录支持状态。引用必须能直接推出对应事实，只有主题相关但不能支持结论的内容不得视为有效证据 |

事实对齐状态定义如下：

- `supported`：证据能够直接支持事实，且金额、日期、对象、适用条件和否定关系一致。
- `partial`：证据只支持事实的一部分，或遗漏适用范围、例外条件等关键限定。
- `unsupported`：最终 Top-5 中没有能够支持该事实的证据。
- `conflict`：检索证据之间或证据与回答之间存在相互冲突的结论。

核验时必须重点比较实体、数字、币种、时间、职级、地域、舱位、酒店标准、审批阈值、否定词和条件范围，不能仅以关键词同时出现作为“已支持”的依据。每条可核验事实的对齐结果写入 `claim_evidence_map`，至少包含：

```text
claim
status
supporting_chunk_ids
evidence_spans
reason
```

输出决策规则固定为：

```text
生成回答草稿（temperature=0.1）
    -> 拆分原子事实并与最终 Top-5 对齐
    -> 全部 supported：输出 rag_grounded
    -> 存在 partial / unsupported / conflict：执行一次自校正
    -> 校正后全部 supported：输出 rag_grounded
    -> 校正后仍未通过：删除制度性结论并输出资料不足回答
```

当证据相互冲突、检索结果不足或二次核验仍失败时，系统不得自行选择一个结论。回答应明确指出资料不足或存在冲突，将 `answer_mode` 切换为 `llm_fallback`，风险等级至少为 `medium`；涉及金额、报销、审批、舱位或酒店标准时提示人工确认。`response_reviewer` 对 RAG 回答所做的任何改写也必须重新经过 `grounding_verifier`，避免复核阶段重新引入无依据陈述。

### 8.8 关键领域对象

- `TravelRequest`：员工、职级、出发地、目的地、日期、目的和偏好舱位。
- `Itinerary`：行程段、预估总额和差标警告。
- `TravelPolicy`：职级舱位、城市酒店上限、日补贴、提前预订天数和事前审批金额线。
- `execution_plan`：Planner 经单次调用生成并通过校验的主意图、任务列表、各任务参数与依赖，以及缺失参数和澄清问题。
- `task_results`：多任务执行结果列表，按任务 ID 关联意图、状态、回答、引用、核验、工具轨迹、用量以及各自的草稿和审批结果。
- `policy_constraints`：制度推理节点输出的来源、置信度、备注，以及酒店限额、提前预订、审批阈值、舱位和火车席别等约束。
- `policy_validation`：约束与草稿的逐项检查结果，状态为 `passed`、`needs_review` 或 `failed`，并分别记录通过项、警告和违规项。
- `ApprovalForm`：是否需要人工审批、风险原因和差标警告。
- `BookingDraft`：由工具轨迹中的 `travel_recommendation` 载荷派生，包含推荐交通/酒店、价格快照、确认项、下一步动作和有效期；上游候选失败时允许生成不完整草稿。
- `ChatResponse`：兼容 Chat Completions 的 `choices`，并扩展表格、引用、执行计划、逐任务结果、制度约束、合规校验、审批、草稿、轨迹和风险字段。

## 9. API 设计

| 方法 | 路径 | 说明 | 关键失败条件 |
| --- | --- | --- | --- |
| GET | `/` | 服务入口、文档和控制台链接 | - |
| POST | `/api/v1/chat` | 非流式 JSON 或 SSE 对话 | LLM/编排异常 |
| GET | `/api/v1/health` | Redis、PostgreSQL、Milvus、关键词索引健康状态 | 已初始化依赖探测失败时返回 degraded 内容 |
| GET | `/api/v1/sessions` | 扫描并列出会话 | Redis 不可用时 503 |
| GET | `/api/v1/sessions/{id}` | 查询会话历史 | Redis 不可用 503；不存在 404 |
| DELETE | `/api/v1/sessions/{id}` | 删除会话 | Redis 不可用时 503 |
| POST | `/api/v1/documents/ingest` | 单段文本入库 | Milvus 不可用 503 |
| POST | `/api/v1/documents/ingest-long` | 长文本分块入库 | 参数不合法 422；写入失败 500 |
| POST | `/api/v1/documents/upload` | 上传 txt/md/html/htm/pdf/docx/pptx/xlsx 并入库；Office 与网页格式优先使用 Unstructured | 格式不支持 415；解析失败 422 |
| GET | `/api/v1/documents` | 列出知识 chunk | Milvus 不可用 503 |
| GET | `/api/v1/documents/search` | 文本混合检索（关键词 + 向量，RRF 后 rerank） | 任一检索通道不可用或检索失败 |
| DELETE | `/api/v1/documents/{id}` | 按父文档或 chunk ID 删除 | Milvus 不可用 503 |
| POST | `/api/v1/documents/batch-delete` | 批量删除文档 | 任一删除异常时 500 |
| DELETE | `/api/v1/documents?title=...` | 按标题删除 | Milvus 不可用 503 |
| POST | `/api/v1/mcp/rpc` | MCP 初始化、工具、资源调用 | JSON-RPC error 响应 |

非流式 `/chat` 返回 OpenAI 风格基础结构，并增加：

- `tables`：从执行计划、制度约束、合规校验、行程文本、库存 JSON、Markdown 表格和预订草稿转换出的结构化表格。
- `citations`：RAG 命中的标题、类型、正文片段和相似度。
- `execution_plan`：统一意图与规划节点生成的已校验任务计划；规划失败或需要澄清时保留对应状态，不生成关键词兜底任务。
- `task_results`：包含 `task_id`、`intent`、`request`、`status`、`answer`、`citations`、`tool_trace`、`verification`、`claim_evidence_map`、`booking_draft`、`approval_form`、`policy_constraints`、`policy_validation`、`risk_level` 和 `usage`，保留每个任务的完整结果及证据归属。
- `route`：`single_task`、`multi_task` 或 `clarification`，说明顶层处理分支；来自编排器内部 `metadata.route`。
- `intent`：本次请求的主意图；实际任务路由以 `tasks[].intent` 为准。
- `policy_constraints`：制度资料中抽取并合并后的结构化约束及其来源、置信度。
- `policy_validation`：自动合规校验状态、检查项、通过项、警告和违规项。
- `approval_form`、`booking_draft`：单任务的人工审批和预订确认数据；多任务时顶层为空，读取对应 `task_results` 中的结果。
- `trace`、`risk_level`、`answer_mode`、`verification`、`claim_evidence_map`：解释、治理、逐条事实-证据对齐和核验信息。

Web 控制台展示结构化任务计划和逐任务执行结果，包含任务状态、回答、引用及审批/草稿信息；轨迹与表格保留制度、合规和节点级执行信息。

## 10. 部署架构

```mermaid
flowchart TB
    Client[Browser / OA / MCP Client]

    subgraph Docker["Docker Compose 网络"]
        App[FastAPI App<br/>:8000]
        Redis[(Redis 7<br/>:6379)]
        PG[(PostgreSQL 16<br/>:5432)]
        Milvus[Milvus Standalone<br/>:19530 / :9091]
        Etcd[(etcd)]
        MinIO[(MinIO<br/>:9000)]

        App --> Redis
        App --> PG
        App --> Milvus
        Milvus --> Etcd
        Milvus --> MinIO
    end

    Client --> App
    App --> ModelAPI[OpenAI-compatible APIs]
    App --> FlyAI[FlyAI CLI / API]
    App --> Railway[12306 Skill / MCP]
```

应用容器内安装 Python 依赖、Node.js、npm 和 `@fly-ai/flyai-cli`。`external/12306` 以可写目录挂载，FlyAI skill 以只读目录挂载。Docker Compose 会等待 PostgreSQL 和 Milvus 健康、Redis 启动后再创建应用容器；应用自身运行时对 Redis/Milvus 连接失败采取降级而不是退出。

### 10.1 启动与关闭

启动生命周期：

1. 配置结构化日志和 OpenTelemetry `TracerProvider`。
2. 尝试连接 Redis；失败时记录 warning 并将客户端置空。
3. 创建 PostgreSQL 异步连接池。
4. 连接 Milvus，并在需要时创建集合和索引。
5. 根据 `AGENT_ORCHESTRATOR_BACKEND` 创建 LangGraph 或 legacy 编排器。

关闭时释放 Redis 客户端和 PostgreSQL 连接池。Milvus 当前没有显式断开逻辑。

## 11. 非功能设计

### 11.1 可用性与降级

- Redis 失败：聊天仍可运行，但会话历史、长期记忆、草稿持久化及会话管理接口不可用。
- Milvus 或关键词索引失败：文档接口返回依赖错误；RAG 对话改用明确标注的 `llm_fallback`，风险等级至少为 medium。
- PostgreSQL 失败：健康状态 degraded；当前聊天主流程不受影响。
- 外部库存失败：返回空结果、错误和免责声明，不伪造实时库存。
- 候选校验失败：对空候选、工具失败、酒店/舱位/席别等可修复问题携带校验反馈自动重试一次；第二次仍失败时停止自动尝试并强制人工审核。
- LangGraph 导入失败：启动时回退到 `TravelOrchestrator`。
- 聊天模型连续失败：`LLMService` 的 circuit breaker 根据阈值在 closed、open、half-open 状态间切换。
- ReAct 超限：返回“达到最大推理轮次”，避免无限工具循环。
- Planner 调用失败、超时、JSON/Schema/依赖校验失败或关键参数缺失：进入澄清分支，不调用业务工具，不再请求一次规划模型，也不默认执行其中一个任务。
- 多任务执行失败或超时：保留其他已完成任务；失败前置任务的后继任务标记为 `blocked`；对所有任务逐项报告状态。
- 制度推理检索失败：输出 `source=retrieval_failed` 的空约束；合规校验将结果标记为需要人工复核，而不是假定符合制度。

当前健康检查将未初始化的 Redis、PostgreSQL 客户端按可用处理；因此启动阶段连接失败未必会直接改变 `status`，这也是后续需要改进的已知边界。

### 11.2 一致性与幂等性

- 客户端重发完整历史时，编排器按首尾重叠消息去重，避免重复持久化。
- 长文档 chunk ID 由父文档 ID、序号和内容哈希稳定生成，同一输入可获得稳定标识。
- Milvus 写入后主动 `flush`，关键词索引同步更新，保证两路检索尽快可见。
- 当前文档批量入库不是事务操作，embedding 成功但 Milvus 或关键词索引部分写入失败时需要运维侧核查并补偿。

### 11.3 性能

- 会话上下文按消息数裁剪；超过阈值时先用 LLM 生成 200 字以内摘要。
- 所有通过护栏的请求只调用一次合并意图与规划的 Planner LLM；任务分流和依赖调度不增加分类模型调用。旅行请求还可能增加制度约束抽取，候选校验失败时还会增加 ReAct 与供应商查询。
- 一个计划最多 8 个业务任务，默认最多并行 3 个独立只读任务。依赖任务顺序就绪，单任务结果隔离；规划默认超时 30 秒，子任务默认超时 120 秒。综合推荐工具内部的库存并发独立于任务级并发，需结合供应商限流配置容量。
- 文档按类型执行语义切分：Markdown 按标题层级、PDF 先解析表格/图片再按段落、代码按函数/类、FAQ 按完整问答对；允许重叠的自然语言 chunk 保留 15% overlap，仅在超出 embedding token 上限时执行类型安全的递归拆分。
- 综合旅行推荐并发查询航班、酒店和火车，减少总等待时间。
- Milvus 使用 IVF_FLAT 索引；数据规模增长后应通过评测调整 `nlist`、`nprobe`、双路召回 Top-20 和最终 Top-5。
- 当前 SSE 不降低首字节等待时间，若需要真正流式体验，应将模型流与工具事件直接转发给客户端。

### 11.4 安全与合规

当前已有控制：

- Pydantic 对 API 请求、工具参数和响应结构进行校验。
- 护栏阻止绕过审批、伪造发票和部分提示注入语句。
- Planner 使用严格 JSON Schema，并在本地检查意图白名单、槽位类型、任务数量、ID 长度、非空文本、唯一任务 ID 和无环依赖；模型不能通过计划指定任意 Python 函数或任意工作流节点。子任务只能进入预定义业务分支，工具调用仍受在线工具白名单及参数校验限制。
- 回答契约要求区分制度引用、工具数据、模型建议和人工待确认项。
- 制度约束只能从检索资料抽取；正则启发式仅识别受支持的酒店限额、提前天数、审批阈值、经济舱和二等座约束。
- 知识库依据不足时不宣称制度结论，并提高风险等级。
- RAG 生成使用受限提示词和 `temperature=0.1`；每条可核验事实必须与最终 Top-5 证据对齐，失败时最多自校正一次，仍失败则回退为资料不足回答。
- 自动流程不会下单；综合推荐成功时只生成草稿，工具轨迹存在且涉及风险时输出 `pending_human_approval`。

生产部署前仍需补齐：

- API 身份认证、企业租户隔离和 RBAC。
- `session_id`、`user_id` 与当前身份的服务端绑定，防止越权读取会话或长期记忆。
- CORS 白名单；当前配置为全来源且允许 credentials，不适合直接暴露到公网。
- 上传大小限制、恶意文件检测、PII 脱敏、日志敏感字段过滤和数据保留策略。
- MCP 调用鉴权、调用方审计、工具级授权和速率限制。
- 供应商密钥使用 Secret Manager 管理，禁止使用示例密钥进入生产。

### 11.5 可观测性与质量

- 已配置结构化日志和基础 OpenTelemetry provider，工作流状态中另有节点级 `trace`。
- 当前未配置 trace exporter、集中式指标后端和统一 request ID，需要在生产环境补充。
- 单元测试覆盖计划 Schema 与依赖校验、单/多任务路由、只读任务并发、依赖顺序、部分失败和阻塞、子任务状态与证据隔离、统一持久化，以及既有制度、合规、MCP、RAG 和旅行搜索能力。
- `evals/build_rag_cases.py` 可从当前知识库生成候选问题与人工复核 CSV；`evals/apply_rag_review.py` 根据复核结果生成精选案例集。
- `evals/run_rag_eval.py` 支持多组 K 值以及仅检索模式，可计算 `hit@k`、`precision@k`、`recall@k`、MRR、检索平均/P95 延迟、关键词准确率和引用准确率。

## 12. 关键配置

| 配置组 | 代表变量 | 作用 |
| --- | --- | --- |
| Chat LLM | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` | 对话、工具选择、摘要、反思 |
| Embedding | `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_DIMENSIONS` | 文档和查询向量化 |
| 存储 | `REDIS_URL`、`DATABASE_URL`、`MILVUS_HOST`、`MILVUS_PORT`、`KEYWORD_INDEX_REDIS_KEY` | 会话、健康探测、向量知识库和关键词索引 |
| RAG | `RAG_KEYWORD_TOP_K`、`RAG_VECTOR_TOP_K`、`RAG_RRF_K`、`RAG_FUSED_TOP_K`、`RAG_FINAL_TOP_K`、`RAG_RERANKER_*` | 双路召回、RRF 融合、候选截断与 Cross-Encoder 重排 |
| 工作流 | `AGENT_ORCHESTRATOR_BACKEND`、`MAX_REACT_ITERATIONS`、`TRAVEL_VALIDATION_MAX_RETRIES` | 编排后端、单轮 ReAct 循环上限和候选合规重试次数（默认 1）；变量名为兼容现有部署保留 `AGENT_` 前缀 |
| 意图与任务规划 | `PLANNER_RESPONSE_FORMAT`、`PLANNER_TIMEOUT_SECONDS` | 默认 `json_schema` 严格输出和 30 秒规划超时；不支持 Schema 的模型供应商可显式使用 `json_object`，仍须通过本地校验 |
| 多任务调度 | `TASK_MAX_CONCURRENCY`、`TASK_TIMEOUT_SECONDS` | 默认并发 3 个独立只读任务，单子任务超时 120 秒；任务数量上限为 Schema 固定的 8 个 |
| 记忆 | `MEMORY_WINDOW_SIZE`、`MEMORY_SUMMARY_THRESHOLD`、`MEMORY_MAX_MESSAGES`、`MEMORY_SESSION_TTL_SECONDS` | 上下文窗口、摘要和持久化 |
| 航班酒店 | `TRAVEL_INVENTORY_PROVIDER`、`FLYAI_*`、`AMADEUS_*` | provider 与凭据 |
| 火车票 | `RAILWAY_12306_SKILL_DIR`、`RAILWAY_MCP_URL`、`RAILWAY_*_TIMEOUT_S` | 本地 Skill 或远程 MCP |
| 熔断 | `CIRCUIT_BREAKER_*` | 聊天模型故障隔离与恢复 |

注意：Milvus 集合当前固定为 1536 维，实际 embedding 模型输出维度必须与其一致。仅修改 `EMBEDDING_DIMENSIONS` 不会自动迁移已有集合。

## 13. 已知技术边界与演进建议

| 优先级 | 当前边界 | 建议 |
| --- | --- | --- |
| P0 | 无认证、租户隔离和权限模型 | 在公网或企业集成前加入 OIDC/JWT、RBAC、租户维度存储键和审计日志 |
| P0 | 预订草稿尚未形成受控状态机 | 建立 draft -> approved -> held -> confirmed/cancelled 状态机，所有外部副作用使用幂等键 |
| P1 | PostgreSQL 未承载业务数据 | 持久化审批、行程、预订、审计事件；Redis 只保留缓存和短期会话 |
| P1 | SSE 为完成后字符回放 | 设计工作流事件协议，实时发送模型 delta、tool_call、tool_result 和 done |
| P1 | 差标规则为代码内默认值 | 将政策版本化并按企业、职级、城市和生效日期管理；计算结果保留 policy_version |
| P1 | 格式感知语义分块和 15% overlap 已接入在线 ETL；Markdown 标题路径、PDF 页码/表格/OCR、FAQ 完整问答对和 Markdown 代码块均能保留 | 增加独立源代码文件白名单及多语言 AST 分块；针对超长代码、复杂表格和非标准 FAQ 建立人工复核与专项评测 |
| P1 | PDF 已接入 PyMuPDF 表格解析、图片统计和低文本密度页面按需 OCR，并保留 OCR 置信度与 `needs_review` | 加强跨页表格、合并单元格、双栏阅读顺序和图片区域级 OCR；用真实制度 PDF 建立解析准确率基线 |
| P1 | 在线 RAG 已接入 Redis BM25 关键词召回、Milvus 向量召回、RRF（`k=60`）、候选 Top-20 和 Cross-Encoder rerank Top-5 | 增加租户/doc_type/生效日期过滤、最低相关性阈值、模型预热和大规模关键词索引方案，持续评估 Redis 全量 BM25 的容量边界 |
| P1 | 已接入 `claim_evidence_map`、四态事实核验、一次有界自校正和二次失败强制回退 | 通过 NLI/LLM-as-judge 与人工标注集提升否定、条件范围、跨句证据和证据冲突的识别准确率 |
| P1 | 已合并意图与规划，通过 Schema 校验后的业务任务与依赖驱动单/多任务分支；缺参或非法计划转澄清 | 建立人工标注的复合请求评测集，衡量任务遗漏、过度拆分、槽位准确率和依赖正确率，再评估经过验证的简单请求快车道 |
| P1 | 多任务结果以隔离状态执行并逐项汇总，只有独立只读任务并行；legacy 回退不具备同等调度保证 | 增加持久化任务状态和中断恢复，统一供应商级并发预算；在有权限及幂等保障后再扩展有外部副作用的业务任务 |
| P1 | 制度约束抽取与校验规则覆盖有限 | 建立版本化约束 schema、单位归一化、城市/职级作用域和冲突解决策略 |
| P1 | 文档写入无事务和任务队列 | 对大文档采用异步任务、状态表、重试与补偿删除 |
| P2 | 部分 `app/core`、`app/infrastructure` 能力与在线链路重复 | 明确唯一编排、LLM、Milvus 抽象，逐步收敛重复实现 |
| P2 | 健康检查将未初始化客户端视为正常 | 区分 disabled、healthy、unhealthy，增加 readiness 与 liveness 两类探针 |
| P2 | 仅有进程内 TraceProvider | 配置 OTLP exporter、指标、日志关联和 SLO 告警 |

## 14. 代码导航

```text
app/
├── main.py                         # 应用工厂与生命周期
├── config.py                       # 环境配置
├── api/routes/                     # REST、SSE、MCP 接口
├── agent/
│   ├── langgraph_orchestrator.py   # 默认主编排图
│   └── orchestrator.py             # 共享能力与 legacy 编排器
├── core/
│   ├── agent/                      # 通用 Planner/ReAct/Reflection
│   ├── intent/                     # 可复用意图组件，非主图分类入口
│   ├── memory/                     # 通用记忆组件
│   ├── rag/                        # 多路检索、重排、生成组件
│   └── tools/                      # 工具注册、库存、MCP 客户端
├── domain/
│   ├── task_plan.py                # 合并意图/任务计划模型与严格 JSON Schema
│   └── travel/                     # 行程和差标领域模型
├── etl/                            # 文档分块与入库管线
├── services/                       # 在线 LLM、Embedding、Milvus 实现
├── infrastructure/                 # 基础设施抽象与可观测性
└── static/                         # 管理与演示控制台

tests/                              # 单元和路由测试，含计划 Schema/任务调度/制度/评测指标
evals/                              # RAG 案例生成、人工复核、精选集和在线评测
external/                           # 12306、FlyAI 外部能力包
docker-compose.yml                  # 本地完整依赖拓扑
Dockerfile                          # 应用镜像
```

## 15. 设计结论

当前系统以 FastAPI 为统一接入层、LangGraph 为默认决策与编排核心、本地领域规则和外部库存工具为行动层、Redis、Milvus 与关键词索引为记忆和知识层。其主要设计特点是：先融合上下文，通过一次 Planner LLM 联合识别意图与规划业务任务，经过 Schema 和依赖校验后按任务数量进入单任务或多任务分支；独立只读任务有界并行，依赖任务顺序执行，逐项保留状态、引用与治理结果并统一持久化；RAG 对问题执行关键词与向量双路 Top-20 召回，使用 RRF（`k=60`）融合后保留候选 Top-20，再经 rerank 返回最终 Top-5；RAG 回答使用受限提示词和 `temperature=0.1`，逐条执行事实-证据对齐，失败时进行一次有界自校正，仍失败则回退为资料不足回答；旅行分支先抽取制度约束，再执行库存工具和自动合规校验，对空候选、工具失败或可修复差标执行一次带反馈的候选重试，第二次仍失败则提升风险并转人工审核；在存在综合推荐结果时生成预订草稿及审批信息；其他非 `rag_grounded` 回答经过同一核验节点时以 `non_rag_answer` 跳过 RAG 事实核验；依赖故障时尽量显式降级而不是伪造结果。

要进入企业生产环境，下一阶段应优先完成身份与租户隔离、审批/预订持久化状态机、真正的流式事件协议、差标规则版本化以及完整的可观测性闭环。

## 16. 修订记录

| 版本 | 日期 | 主要变更 |
| --- | --- | --- |
| 1.7 | 2026-09-28 | 合并意图识别与任务规划为单次结构化模型调用；增加计划校验、单/多任务分流、依赖调度、逐任务结果和统一持久化；明确规划失败与缺参澄清，不再使用关键词兜底计划 |
| 1.6 | 2026-09-22 | 双路混合检索、RRF（`k=60`）、Top-20 rerank 至 Top-5、格式感知语义分块、PDF/OCR 以及事实核验与自校正 |
