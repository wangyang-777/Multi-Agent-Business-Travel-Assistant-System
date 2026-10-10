const state = {
  view: "chat",
  sessionId: (localStorage.getItem("travelAgentSessionId") || "demo-1").trim() || "demo-1",
  chatHistory: [],
  lastResponse: null,
  docs: [],
  selectedDocIds: new Set(),
  sessions: [],
  selectedSession: null,
  health: null,
  reviewToken: "",
  activeReview: null,
  sessionReviews: [],
};

const evalCases = [
  {
    id: "hotel-tier1-staff",
    question: "staff 去上海出差酒店标准是多少？",
    expected: ["policy-hotel-tier1"],
    keywords: ["800", "CNY", "酒店"],
  },
  {
    id: "advance-booking",
    question: "公司要求提前几天预订差旅行程？",
    expected: ["policy-advance-booking"],
    keywords: ["提前", "7", "天"],
  },
  {
    id: "staff-cabin",
    question: "staff 出差可以选择什么舱位？",
    expected: ["policy-cabin-staff"],
    keywords: ["staff", "经济舱"],
  },
  {
    id: "approval-threshold",
    question: "差旅金额超过多少需要事前审批？",
    expected: ["policy-approval-threshold"],
    keywords: ["5000", "审批"],
  },
];

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const error = new Error(data.detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

async function apiForm(path, formData) {
  const response = await fetch(path, { method: "POST", body: formData });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const error = new Error(data.detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function setView(view) {
  state.view = view;
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $$(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
  const titles = {
    chat: ["Chat 工作台", "差旅规划、差标校验、审批提示与引用来源"],
    history: ["History", "按 session_id 查看、恢复与管理历史对话"],
    knowledge: ["Knowledge", "政策文档入库、查看与删除"],
    eval: ["RAG 评测", "检索命中、回答关键词与引用准确率"],
    reviews: ["人工审核", "核对制度依据，确认条件并发布审核结论"],
    system: ["System", "运行依赖与服务健康状态"],
  };
  $("#view-title").textContent = titles[view][0];
  $("#view-subtitle").textContent = titles[view][1];
  if (view === "chat") loadSessionOptions();
  if (view === "history") loadSessions();
  if (view === "knowledge") loadDocuments();
  if (view === "system") loadHealth();
}

function syncSessionInput() {
  const input = $("#session-id");
  state.sessionId = (state.sessionId || "demo-1").trim() || "demo-1";
  if (input) input.value = state.sessionId;
}

function renderRecentSessions() {
  const select = $("#recent-sessions");
  if (!select) return;
  const current = select.value;
  select.innerHTML = `<option value="">选择历史会话</option>` + state.sessions
    .map((item) => `<option value="${escapeHtml(item.session_id)}">${escapeHtml(item.session_id)} · ${escapeHtml(item.message_count)}条</option>`)
    .join("");
  select.value = current || "";
}

async function loadSessionOptions() {
  try {
    const data = await api("/api/v1/sessions?limit=50");
    state.sessions = data.sessions || [];
    renderRecentSessions();
  } catch {
    state.sessions = [];
    renderRecentSessions();
  }
}

function renderMessages() {
  const box = $("#messages");
  if (!state.chatHistory.length) {
    box.innerHTML = `<div class="empty">输入差旅需求或制度问题后，这里会显示对话。</div>`;
    renderResponseDetails(null);
    return;
  }
  box.innerHTML = state.chatHistory
    .map((msg) => `<div class="message ${msg.role}">${escapeHtml(msg.content)}</div>`)
    .join("");
  box.scrollTop = box.scrollHeight;
  renderResponseDetails(state.lastResponse);
}

function renderResponseDetails(response) {
  if (response?.human_reviews?.length) {
    const storageKey = `travelAgentReviews:${state.sessionId}`;
    let saved = [];
    try { saved = JSON.parse(localStorage.getItem(storageKey) || "[]"); } catch { saved = []; }
    const refs = new Map(saved.map((item) => [item.review_id, item]));
    for (const item of response.human_reviews) if (item.review_id && item.result_token) refs.set(item.review_id, { review_id: item.review_id, result_token: item.result_token });
    localStorage.setItem(storageKey, JSON.stringify([...refs.values()].slice(-100)));
  }
  renderHumanReviewSummaries(response?.human_reviews || state.sessionReviews || []);
  renderTables(response?.tables || []);
  renderApproval(response?.approval_form || null, response?.risk_level || null, response?.booking_draft || null);
  renderCitations(response?.citations || [], response?.answer_mode || null);
  renderTrace(
    response?.trace || [],
    response?.answer_mode || null,
    response?.verification || null,
    response?.execution_plan || null,
    response?.policy_constraints || null,
    response?.policy_validation || null,
  );
}

function renderTables(tables) {
  const target = $("#tab-tables");
  if (!tables.length) {
    target.innerHTML = `<div class="empty">暂无表格化结果。</div>`;
    return;
  }
  target.innerHTML = tables
    .map((table) => {
      const head = table.columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
      const rows = table.rows
        .map((row) => `<tr>${table.columns.map((col) => `<td>${escapeHtml(row[col] || "")}</td>`).join("")}</tr>`)
        .join("");
      return `<div class="table-title">${escapeHtml(table.title)}</div><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
    })
    .join("");
}

function renderApproval(form, riskLevel, bookingDraft) {
  const target = $("#tab-approval");
  const draftHtml = renderBookingDraft(bookingDraft);
  if (!form) {
    target.innerHTML = `<div class="empty">本次回答未生成审批单。${riskLevel ? `当前风险等级：${escapeHtml(riskLevel)}。` : ""}</div>`;
    if (draftHtml) target.innerHTML += draftHtml;
    return;
  }
  const rows = [
    ["风险等级", riskLevel || "-"],
    ["审批状态", form.status],
    ["是否需要审批", form.required ? "是" : "否"],
    ["员工", form.employee_id || "-"],
    ["职级", form.grade || "-"],
    ["路线", `${form.origin_city || "-"} → ${form.destination_city || "-"}`],
    ["出发日期", form.departure_date || "-"],
    ["预估金额", form.estimated_total_cny || "-"],
  ];
  const warnings = (form.policy_warnings || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  target.innerHTML = `<div class="kv">${rows.map(([k, v]) => `<span>${escapeHtml(k)}</span><strong>${escapeHtml(v)}</strong>`).join("")}</div>${warnings ? `<div class="item-title">提示</div><ul>${warnings}</ul>` : ""}${draftHtml}`;
}

function renderBookingDraft(draft) {
  if (!draft) return "";
  const flight = draft.recommended_flight || {};
  const train = draft.recommended_train || {};
  const hotel = draft.recommended_hotel || {};
  const rows = [
    ["草稿 ID", draft.draft_id || "-"],
    ["草稿状态", draft.status || "-"],
    ["下一步", draft.next_action || "-"],
    ["推荐航班", flight.flight_no || "-"],
    ["推荐高铁/火车", train.train_code || "-"],
    ["推荐酒店", hotel.name || "-"],
    ["预估金额", draft.estimated_total_cny || "待确认"],
    ["有效期至", draft.expires_at || "-"],
  ];
  const items = (draft.confirmation_items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  return `<div class="item-title">预订草稿</div><div class="kv">${rows.map(([k, v]) => `<span>${escapeHtml(k)}</span><strong>${escapeHtml(v)}</strong>`).join("")}</div>${items ? `<div class="item-title">确认项</div><ul>${items}</ul>` : ""}`;
}

function renderCitations(citations, answerMode) {
  const target = $("#tab-citations");
  if (!citations.length) {
    const hint = answerMode === "llm_fallback"
      ? "当前回答为 LLM fallback 或实时查询结果，未使用可展示的 RAG 引用。"
      : "当前回答未返回可展示的 RAG 引用。";
    target.innerHTML = `<div class="empty">${escapeHtml(hint)}</div>`;
    return;
  }
  target.innerHTML = citations
    .map(
      (item, index) => `<div class="citation"><div class="item-title">[${index + 1}] ${escapeHtml(item.title || "未命名")}</div><div>${escapeHtml(item.content || "")}</div><div class="doc-meta"><span>${escapeHtml(item.doc_type || "-")}</span><span>score: ${escapeHtml(item.score ?? "-")}</span></div></div>`,
    )
    .join("");
}

function compactJson(value) {
  return escapeHtml(JSON.stringify(value, null, 2));
}

function renderTrace(trace, answerMode, verification, executionPlan, policyConstraints, policyValidation) {
  const target = $("#tab-trace");
  if (!trace.length && !answerMode && !verification && !executionPlan && !policyConstraints && !policyValidation) {
    target.innerHTML = `<div class="empty">暂无执行轨迹。</div>`;
    return;
  }
  const mode = answerMode
    ? `<div class="trace-item"><strong>answer_mode</strong><div>${escapeHtml(answerMode === "rag_grounded" ? "有制度依据" : "模型建议")}</div></div>`
    : "";
  const verify = verification
    ? `<div class="trace-item"><strong>verification</strong><div class="${verification.passed ? "pass" : "fail"}">${verification.passed ? "已通过事实核验" : "存在未核验事实"}</div><div>${escapeHtml((verification.unsupported_terms || []).join("；"))}</div></div>`
    : "";
  const plan = executionPlan
    ? `<div class="trace-item"><strong>execution_plan</strong><pre class="json-block">${compactJson(executionPlan)}</pre></div>`
    : "";
  const constraints = policyConstraints
    ? `<div class="trace-item"><strong>policy_constraints</strong><pre class="json-block">${compactJson(policyConstraints)}</pre></div>`
    : "";
  const validation = policyValidation
    ? `<div class="trace-item"><strong>policy_validation</strong><pre class="json-block">${compactJson(policyValidation)}</pre></div>`
    : "";
  target.innerHTML = mode + verify + plan + constraints + validation + trace
    .map((item) => `<div class="trace-item"><strong>${escapeHtml(item.node)}</strong><div>${escapeHtml(item.event)}</div></div>`)
    .join("");
}

async function sendChat(event) {
  event.preventDefault();
  const sessionValue = $("#session-id").value.trim();
  if (sessionValue) {
    state.sessionId = sessionValue;
    localStorage.setItem("travelAgentSessionId", state.sessionId);
  }
  const input = $("#chat-input");
  const content = input.value.trim();
  if (!content) return;
  input.value = "";
  state.chatHistory.push({ role: "user", content });
  renderMessages();
  $("#send-btn").disabled = true;
  try {
    const response = await api("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ messages: state.chatHistory, stream: false, session_id: state.sessionId }),
    });
    const answer = response.choices?.[0]?.message?.content || "";
    state.chatHistory.push({ role: "assistant", content: answer });
    state.lastResponse = response;
  } catch (error) {
    state.chatHistory.push({ role: "assistant", content: `请求失败：${error.message}` });
    state.lastResponse = null;
  } finally {
    $("#send-btn").disabled = false;
    renderMessages();
  }
}

async function loadCurrentSession() {
  state.sessionReviews = [];
  const sessionId = $("#session-id").value.trim();
  if (!sessionId) {
    await loadSessionOptions();
    state.chatHistory = [{ role: "assistant", content: "请先从“最近会话”选择一个 session_id，或手动输入 session_id。" }];
    state.lastResponse = null;
    renderMessages();
    return;
  }
  state.sessionId = sessionId;
  localStorage.setItem("travelAgentSessionId", state.sessionId);
  try {
    const data = await api(`/api/v1/sessions/${encodeURIComponent(state.sessionId)}`);
    state.chatHistory = data.messages || [];
    state.lastResponse = null;
  } catch (error) {
    state.chatHistory = [{ role: "assistant", content: `未加载到该 session 历史：${error.message}` }];
    state.lastResponse = null;
  }
  renderMessages();
  refreshSessionReviews();
}

async function refreshSessionReviews() {
  try {
    const refs = JSON.parse(localStorage.getItem(`travelAgentReviews:${state.sessionId}`) || "[]");
    const reviews = await Promise.all(refs.map(async (ref) => {
      try {
        const result = await api(`/api/v1/human-reviews/${encodeURIComponent(ref.review_id)}/result?session_id=${encodeURIComponent(state.sessionId)}`, { headers: { "X-Review-Result-Token": ref.result_token } });
        return { ...result, result_token: ref.result_token };
      } catch (error) {
        if (error.status === 404) return { ...ref, status: "expired", reasons: [{ detail: "审核单已过期或回执凭据无效，请联系审核人员。" }] };
        throw error;
      }
    }));
    state.sessionReviews = reviews;
    if (state.lastResponse) state.lastResponse.human_reviews = state.sessionReviews;
    renderHumanReviewSummaries(state.sessionReviews);
  } catch (error) {
    $("#tab-review").textContent = `查询审核结果失败：${error.message}`;
  }
}

function undoLastTurn() {
  if (!state.chatHistory.length) return;
  if (state.chatHistory.at(-1)?.role === "assistant") state.chatHistory.pop();
  if (state.chatHistory.at(-1)?.role === "user") state.chatHistory.pop();
  state.lastResponse = null;
  renderMessages();
}

async function loadDocuments() {
  const target = $("#doc-list");
  target.innerHTML = `<div class="empty">Loading...</div>`;
  try {
    const data = await api("/api/v1/documents?limit=200");
    state.docs = data.documents || [];
    state.selectedDocIds = new Set([...state.selectedDocIds].filter((id) => state.docs.some((doc) => doc.id === id)));
    updateDocSelectionStatus();
    if (!state.docs.length) {
      target.innerHTML = `<div class="empty">知识库为空。</div>`;
      return;
    }
    target.innerHTML = state.docs
      .map(
        (doc) => `<div class="doc-item"><div class="panel-head"><label class="doc-select"><input type="checkbox" data-select-doc="${escapeHtml(doc.id)}" ${state.selectedDocIds.has(doc.id) ? "checked" : ""} /><span><strong>${escapeHtml(doc.title)}</strong><div class="doc-meta"><span>${escapeHtml(doc.doc_type)}</span><span>${escapeHtml(doc.id)}</span></div></span></label><button class="danger" data-delete-doc="${escapeHtml(doc.id)}">删除</button></div><div>${escapeHtml(doc.content)}</div></div>`,
      )
      .join("");
  } catch (error) {
    target.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function updateDocSelectionStatus() {
  const target = $("#doc-list-status");
  if (!target) return;
  const total = state.docs.length;
  const selected = state.selectedDocIds.size;
  target.innerHTML = total
    ? `<span>已选择 ${selected} / ${total} 条知识</span>`
    : "";
}

function toggleAllDocuments() {
  if (!state.docs.length) return;
  const allSelected = state.docs.every((doc) => state.selectedDocIds.has(doc.id));
  state.selectedDocIds = allSelected ? new Set() : new Set(state.docs.map((doc) => doc.id));
  $("#doc-list").querySelectorAll("[data-select-doc]").forEach((item) => {
    item.checked = state.selectedDocIds.has(item.dataset.selectDoc);
  });
  updateDocSelectionStatus();
}

async function loadSessions() {
  const target = $("#session-list");
  const detail = $("#history-messages");
  target.innerHTML = `<div class="empty">Loading...</div>`;
  if (!state.selectedSession) {
    detail.innerHTML = `<div class="empty">选择一个 session 查看完整历史。</div>`;
  }
  try {
    const data = await api("/api/v1/sessions?limit=200");
    state.sessions = data.sessions || [];
    renderRecentSessions();
    if (!state.sessions.length) {
      target.innerHTML = `<div class="empty">暂无会话历史。需要先在 Chat 中带 session_id 对话。</div>`;
      return;
    }
    target.innerHTML = state.sessions
      .map(
        (item) => `<button class="session-item ${item.session_id === state.selectedSession?.session_id ? "active" : ""}" data-session-id="${escapeHtml(item.session_id)}"><div><strong>${escapeHtml(item.session_id)}</strong><span>${escapeHtml(item.message_count)} 条消息 · TTL ${escapeHtml(item.ttl_seconds)}s</span></div><p>${escapeHtml(item.last_content || "无内容")}</p></button>`,
      )
      .join("");
  } catch (error) {
    target.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

async function loadSessionDetail(sessionId) {
  const target = $("#history-messages");
  target.innerHTML = `<div class="empty">Loading...</div>`;
  try {
    const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
    state.selectedSession = data;
    $("#history-title").textContent = `历史详情：${data.session_id}`;
    if (!(data.messages || []).length) {
      target.innerHTML = `<div class="empty">该 session 没有消息。</div>`;
      return;
    }
    target.innerHTML = data.messages
      .map((msg) => `<div class="message ${escapeHtml(msg.role)}">${escapeHtml(msg.content)}</div>`)
      .join("");
  } catch (error) {
    state.selectedSession = null;
    target.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function useSelectedSession() {
  if (!state.selectedSession) return;
  state.sessionId = state.selectedSession.session_id;
  state.chatHistory = state.selectedSession.messages || [];
  state.lastResponse = null;
  localStorage.setItem("travelAgentSessionId", state.sessionId);
  syncSessionInput();
  renderMessages();
  setView("chat");
}

async function ingestDocument(event) {
  event.preventDefault();
  const status = $("#doc-status");
  const submit = $("#doc-submit");
  const longMode = $("#doc-long-mode").checked;
  const file = $("#doc-file").files?.[0] || null;
  const payload = {
    title: $("#doc-title").value.trim(),
    doc_type: $("#doc-type").value,
    content: $("#doc-content").value.trim(),
    metadata: {},
  };
  if (!payload.title && !file) {
    status.innerHTML = `<span class="fail">请填写标题或选择文件。</span>`;
    return;
  }
  if (!payload.content && !file) {
    status.innerHTML = `<span class="fail">请粘贴内容或选择文件。</span>`;
    return;
  }
  if (longMode) {
    payload.chunk_size = Number($("#doc-chunk-size").value || 3000);
    payload.chunk_overlap = Math.round(payload.chunk_size * 0.15);
  }
  submit.disabled = true;
  status.innerHTML = `<span>正在解析、分块并写入向量库，长 PDF 可能需要几十秒...</span>`;
  try {
    let result;
    if (file) {
      const form = new FormData();
      form.append("file", file);
      form.append("title", payload.title || file.name);
      form.append("doc_type", payload.doc_type);
      form.append("chunk_size", String(payload.chunk_size || 3000));
      form.append("chunk_overlap", String(payload.chunk_overlap || 450));
      result = await apiForm("/api/v1/documents/upload", form);
    } else {
      const path = longMode ? "/api/v1/documents/ingest-long" : "/api/v1/documents/ingest";
      result = await api(path, { method: "POST", body: JSON.stringify(payload) });
    }
    status.innerHTML = `<span class="pass">入库成功：${escapeHtml(result.chunk_count || 1)} 个 chunk，doc_id=${escapeHtml(result.doc_id)}</span>`;
    $("#doc-form").reset();
    $("#doc-chunk-size").value = "3000";
    $("#doc-chunk-overlap").value = "450";
    await loadDocuments();
  } catch (error) {
    status.innerHTML = `<span class="fail">入库失败：${escapeHtml(error.message)}</span>`;
  } finally {
    submit.disabled = false;
  }
}

async function deleteDocument(id) {
  await api(`/api/v1/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
  state.selectedDocIds.delete(id);
  await loadDocuments();
}

async function deleteSelectedDocuments() {
  const ids = [...state.selectedDocIds];
  const status = $("#doc-list-status");
  if (!ids.length) {
    status.innerHTML = `<span class="fail">请先勾选需要删除的知识。</span>`;
    return;
  }
  const confirmed = window.confirm(`确认删除已选 ${ids.length} 条知识及其 Milvus 向量吗？`);
  if (!confirmed) return;
  status.innerHTML = `<span>正在批量删除 ${ids.length} 条知识...</span>`;
  try {
    const result = await api("/api/v1/documents/batch-delete", {
      method: "POST",
      body: JSON.stringify({ doc_ids: ids }),
    });
    state.selectedDocIds.clear();
    status.innerHTML = `<span class="pass">批量删除成功：删除 ${escapeHtml(result.deleted)} 条向量记录。</span>`;
    await loadDocuments();
  } catch (error) {
    status.innerHTML = `<span class="fail">批量删除失败：${escapeHtml(error.message)}</span>`;
  }
}

function matchesExpected(item, expected) {
  const values = [item.id, item.doc_id, item.title, item.content].filter(Boolean).map(String);
  return expected.some((key) => key && values.some((value) => value.includes(key)));
}

async function runEval() {
  const metrics = $("#eval-metrics");
  const casesBox = $("#eval-cases");
  metrics.innerHTML = `<div class="empty">Running...</div>`;
  casesBox.innerHTML = "";
  const rows = [];
  for (const item of evalCases) {
    const search = await api(`/api/v1/documents/search?q=${encodeURIComponent(item.question)}&top_k=5`);
    const results = search.results || [];
    const rankIndex = results.findIndex((result) => matchesExpected(result, item.expected));
    const rank = rankIndex >= 0 ? rankIndex + 1 : null;
    const chat = await api("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: item.question }], stream: false }),
    });
    const answer = chat.choices?.[0]?.message?.content || "";
    const keywordOk = item.keywords.every((keyword) => answer.toLowerCase().includes(keyword.toLowerCase()));
    const citationOk = (chat.citations || []).some((citation) => matchesExpected(citation, item.expected));
    rows.push({ ...item, rank, hit: Boolean(rank), keywordOk, citationOk, answer });
  }
  const total = rows.length || 1;
  const hit = rows.filter((row) => row.hit).length / total;
  const mrr = rows.reduce((sum, row) => sum + (row.rank ? 1 / row.rank : 0), 0) / total;
  const keyword = rows.filter((row) => row.keywordOk).length / total;
  const citation = rows.filter((row) => row.citationOk).length / total;
  metrics.innerHTML = [
    ["Hit@5", hit],
    ["MRR", mrr],
    ["Keyword", keyword],
    ["Citation", citation],
  ]
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${Number(value).toFixed(2)}</strong></div>`)
    .join("");
  casesBox.innerHTML = rows
    .map(
      (row) => `<div class="eval-row"><div class="panel-head"><strong>${escapeHtml(row.id)}</strong><span class="${row.hit ? "pass" : "fail"}">rank: ${row.rank || "-"}</span></div><div>${escapeHtml(row.answer.slice(0, 180))}</div><div class="doc-meta"><span>keyword: ${row.keywordOk ? "pass" : "fail"}</span><span>citation: ${row.citationOk ? "pass" : "fail"}</span></div></div>`,
    )
    .join("");
}

async function loadHealth() {
  try {
    state.health = await api("/api/v1/health");
    const checks = state.health.checks || {};
    $("#status-strip").innerHTML = `<span class="status-dot"></span><span>Redis ${checks.redis ? "✓" : "!"} · Milvus ${checks.milvus ? "✓" : "!"} · DB ${checks.database ? "✓" : "!"}</span>`;
    $("#system-grid").innerHTML = Object.entries(checks)
      .map(([key, value]) => `<div class="system-cell"><span>${escapeHtml(key)}</span><strong class="${value ? "pass" : "fail"}">${value ? "OK" : "FAIL"}</strong></div>`)
      .join("");
  } catch (error) {
    $("#status-strip").innerHTML = `<span class="status-dot warn"></span><span>Health failed</span>`;
    $("#system-grid").innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
  $$(".tab").forEach((item) => {
    item.addEventListener("click", () => {
      $$(".tab").forEach((tab) => tab.classList.toggle("active", tab === item));
      $$(".tab-page").forEach((page) => page.classList.toggle("active", page.id === `tab-${item.dataset.tab}`));
    });
  });
  $("#chat-form").addEventListener("submit", sendChat);
  $("#session-id").addEventListener("change", () => {
    const value = $("#session-id").value.trim();
    if (!value) return;
    state.sessionId = value;
    localStorage.setItem("travelAgentSessionId", state.sessionId);
  });
  $("#recent-sessions").addEventListener("change", () => {
    const value = $("#recent-sessions").value.trim();
    if (!value) return;
    state.sessionId = value;
    localStorage.setItem("travelAgentSessionId", state.sessionId);
    syncSessionInput();
    loadCurrentSession();
  });
  $("#load-session-btn").addEventListener("click", loadCurrentSession);
  $("#undo-btn").addEventListener("click", undoLastTurn);
  $("#clear-btn").addEventListener("click", () => {
    state.chatHistory = [];
    state.lastResponse = null;
    renderMessages();
  });
  $("#refresh-docs").addEventListener("click", loadDocuments);
  $("#select-all-docs").addEventListener("click", toggleAllDocuments);
  $("#delete-selected-docs").addEventListener("click", deleteSelectedDocuments);
  $("#refresh-sessions").addEventListener("click", loadSessions);
  $("#session-list").addEventListener("click", (event) => {
    const item = event.target.closest("[data-session-id]");
    if (item) loadSessionDetail(item.dataset.sessionId);
  });
  $("#use-session-btn").addEventListener("click", useSelectedSession);
  $("#doc-form").addEventListener("submit", ingestDocument);
  $("#doc-list").addEventListener("click", (event) => {
    const selectedId = event.target?.dataset?.selectDoc;
    if (selectedId) {
      if (event.target.checked) state.selectedDocIds.add(selectedId);
      else state.selectedDocIds.delete(selectedId);
      updateDocSelectionStatus();
      return;
    }
    const id = event.target?.dataset?.deleteDoc;
    if (id) deleteDocument(id);
  });
  $("#review-login").addEventListener("submit", (event) => {
    event.preventDefault();
    state.reviewToken = $("#review-token").value.trim();
    loadReviewQueue();
  });
  $("#review-list").addEventListener("click", (event) => {
    const item = event.target.closest("[data-review-id]");
    if (item) loadReviewDetail(item.dataset.reviewId);
  });
  $("#tab-review").addEventListener("click", (event) => {
    if (event.target.closest("[data-refresh-reviews]")) refreshSessionReviews();
  });
  $("#run-eval").addEventListener("click", runEval);
  $("#refresh-health").addEventListener("click", loadHealth);
}

bindEvents();
syncSessionInput();
renderMessages();
loadHealth();
loadSessionOptions();

function reviewStatus(status) {
  return { pending: "待人工审核", approved: "审核通过", rejected: "审核未通过", needs_information: "需补充信息", submission_failed: "审核单提交失败", expired: "审核回执不可用" }[status] || status;
}

function renderHumanReviewSummaries(reviews) {
  const target = $("#tab-review");
  target.innerHTML = '<button class="ghost" data-refresh-reviews>刷新当前会话审核结果</button>';
  if (!reviews.length) {
    target.innerHTML += '<div class="empty">本次没有待人工审核的问题。</div>';
    return;
  }
  target.innerHTML += reviews.map((review) => `<div class="citation">
    <div class="item-title">${escapeHtml(reviewStatus(review.status))}</div>
    <div>${escapeHtml(review.question || "")}</div>
    <ul>${(review.reasons || []).map((reason) => `<li>${escapeHtml(reason.detail)}</li>`).join("")}</ul>
    ${review.status === "approved" ? `<div class="message assistant">${escapeHtml(review.final_answer || "")}</div>` : ""}
    ${review.notes ? `<p>审核说明：${escapeHtml(review.notes)}</p>` : ""}
    ${review.reviewer ? `<p>审核人：${escapeHtml(review.reviewer)} · ${escapeHtml(review.reviewed_at)}</p>` : ""}
    ${review.status === "needs_information" ? '<p>请补充上述信息后重新提问。</p>' : ""}
    ${review.review_id ? `<small>审核单：${escapeHtml(review.review_id)}</small>` : ""}
  </div>`).join("");
}

async function reviewApi(path, options = {}) {
  return api(path, { ...options, headers: { ...(options.headers || {}), Authorization: `Bearer ${state.reviewToken}` } });
}

async function loadReviewQueue() {
  const target = $("#review-list");
  try {
    const data = await reviewApi("/api/v1/human-reviews");
    $("#review-actor").textContent = `当前审核身份：${data.reviewer}`;
    target.innerHTML = data.reviews.length ? data.reviews.map((ticket) => `<button class="doc-item" data-review-id="${escapeHtml(ticket.review_id)}">${escapeHtml(ticket.task_request || ticket.question)}<br><small>${escapeHtml(ticket.created_at)}</small></button>`).join("") : '<div class="empty">暂无待审核问题。</div>';
  } catch (error) {
    target.textContent = `加载失败：${error.message}`;
  }
}

async function loadReviewDetail(id) {
  const target = $("#review-detail");
  try {
    const ticket = await reviewApi(`/api/v1/human-reviews/${encodeURIComponent(id)}`);
    state.activeReview = ticket;
    const snapshot = ticket.snapshot;
    target.innerHTML = `<h3>${escapeHtml(reviewStatus(ticket.status))}</h3>
      <p>原问题：${escapeHtml(snapshot.question)}</p><p>本次审核范围：${escapeHtml(snapshot.task_request)}</p>
      <h4>待确认事项</h4><ul>${ticket.reasons.map((reason) => `<li>${escapeHtml(reason.detail)}</li>`).join("")}</ul>
      <h4>候选答复（尚未审核）</h4><div class="message assistant">${escapeHtml(snapshot.candidate_answer)}</div>
      <h4>原文依据</h4>${snapshot.citations.map((citation, index) => `<div class="citation"><strong>[${index + 1}] ${escapeHtml(citation.title || "制度原文")}</strong><div class="message">${escapeHtml(citation.content)}</div></div>`).join("")}
      <details><summary>事实、计算和核验记录</summary><pre>${escapeHtml(JSON.stringify({ evidence: snapshot.evidence, verification: snapshot.verification, stages: snapshot.stages }, null, 2))}</pre></details>
      ${ticket.status === "pending" ? `<form class="stack-form" id="review-decision-form">
        <label>审核决定<select id="review-action"><option value="approved">通过并发布答复</option><option value="needs_information">要求补充信息</option><option value="rejected">不通过</option></select></label>
        <label>最终答复<textarea id="review-final-answer" rows="5" placeholder="通过审核时填写正式答复；未通过时留空"></textarea></label>
        <label>审核理由或需补充的信息<textarea id="review-notes" rows="3" required></textarea></label>
        <button class="primary" type="submit" id="review-submit">提交审核决定</button>
      </form>` : `<div class="message assistant">${escapeHtml(ticket.decision?.final_answer || ticket.decision?.notes || "")}</div>`}`;
    $("#review-decision-form")?.addEventListener("submit", submitReviewDecision);
  } catch (error) {
    target.textContent = `加载失败：${error.message}`;
  }
}

async function submitReviewDecision(event) {
  event.preventDefault();
  const button = $("#review-submit");
  button.disabled = true;
  try {
    const id = state.activeReview.review_id;
    const result = await reviewApi(`/api/v1/human-reviews/${encodeURIComponent(id)}/decision`, {
      method: "POST", body: JSON.stringify({ action: $("#review-action").value, final_answer: $("#review-final-answer").value, notes: $("#review-notes").value }),
    });
    $("#review-detail").textContent = `${reviewStatus(result.status)}。${result.final_answer || result.notes}`;
    await loadReviewQueue();
  } catch (error) {
    button.disabled = false;
    const notice = document.createElement("p");
    notice.textContent = `提交失败：${error.message}`;
    $("#review-decision-form").append(notice);
  }
}
