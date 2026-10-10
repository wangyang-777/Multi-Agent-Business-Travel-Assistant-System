# RAG 效果量化

评测分开记录检索、证据覆盖和最终回答。检索命中不能代替答案正确性；关键词包含率只是自动初筛，正式答案分数需要逐题人工核对，特别是制度版本、例外和无法回答的问题。

## 真实制度试点数据集

`real_policy_cases.jsonl` 基于用户提供的 7 页《天津中绿电投资股份有限公司差旅费管理办法》构建，共 45 题：40 题有文档答案、5 题应指出文档未给出信息。题目覆盖适用范围、交通等级、住宿表格、旺季、补助、培训、调动、审批和边界条件。每题含预期答案、证据页码与片段；`relevance_groups` 将等价来源标为同一证据组。需多个事实的题，每个证据组都要找到至少一个来源。

原 PDF 不提交到 Git。把用户提供的 PDF 放到 `evals/source_documents/1219916518.pdf`。构建脚本会校验 SHA256，防止将另一版本的文件误配到当前题集。PDF 元数据为 2024 年 4 月创建，但办法只写“董事会审议通过之日起生效”，没有给出具体通过日期；文件是否仍为最新生效版本、是否存在各单位更低标准，需要制度负责人确认。因此本题集目前标记为 `pending_policy_owner`，报告应称为**单文件试点评测**。

当前 PDF 入库时使用 parent ID `tjzld-travel-policy-2024`、`chunk_size=3000` 和 `RAG_SEMANTIC_PDF_ENABLED=true`。正文先按条款聚合，仅对超过 `RAG_SEMANTIC_MAX_CHARS` 的条款计算相邻句子的 Embedding 余弦相似度，选择低相似度断点；表格保留整表并追加带表头和对应条款语境的逐行索引。PDF 路径不使用固定字符重叠。`real_policy_cases.jsonl` 中的 chunk ID 是旧分块版本的标签；每次重建知识库后应根据证据引文重标当前 ID，不能直接拿旧 ID 评测。

```bash
.venv/bin/python -m evals.build_tjzld_policy_cases

# 入库或重建后，把源证据映射到当前索引的 chunk ID。
.venv/bin/python -m evals.relabel_policy_cases \
  --cases evals/real_policy_cases.jsonl \
  --output evals/runs/real-policy-cases-semantic.jsonl

.venv/bin/python -m evals.run_rag_eval \
  --cases evals/runs/real-policy-cases-semantic.jsonl \
  --top-ks 1,3,5 --skip-chat \
  > evals/runs/real-policy-retrieval-semantic.json
```

构建脚本同时写出 `evals/runs/real-policy-case-review.csv`。制度负责人逐行核对问题、标准答案、页码和引文，在 `reviewer_decision` 填 `keep`、`fix` 或 `drop`；如修正，填写 `corrected_answer` 等列。完成后生成获批题集：

```bash
.venv/bin/python -m evals.apply_rag_review \
  --cases evals/real_policy_cases.jsonl \
  --review-csv evals/runs/real-policy-case-review.csv \
  --output evals/runs/real-policy-cases-approved.jsonl
```

空白决定不会进入获批题集。正式报告使用获批题集，保留原始 PDF、审批记录和对应报告；本次试点运行使用的是待复核题集。

服务需在 `http://127.0.0.1:8000` 运行，并已入库上述 PDF。入库命令：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F 'file=@evals/source_documents/1219916518.pdf' \
  -F 'title=天津中绿电投资股份有限公司差旅费管理办法（待确认生效版本）' \
  -F 'doc_type=policy' \
  -F 'doc_id=tjzld-travel-policy-2024' \
  -F 'chunk_size=3000'
```

同一 parent ID 重复入库前应先删除旧文档，避免 Milvus 产生重复主键记录。

本单文件试点的 45 题检索对照：旧版 10 块与当前 65 块在 40 道有答案题上均达到 `Hit@1=100%`、`all_evidence@5=100%`。当前 65 块中包含 34 个未进一步拆分的条款块、2 个 Embedding 语义断点生成的块、3 个完整表格块、26 个表格行块。原结果见 `evals/runs/real-policy-retrieval.json`，新结果见 `evals/runs/real-policy-retrieval-semantic.json`。这些题目用于开发调试且仍待制度负责人复核，100% 检索分数不代表最终回答准确率或新切块策略在其他制度上的收益。

同一组 12 道难题的初步答案复核：旧版有答案题答对 2/9，当前答对 3/9；两个版本的 3 道无答案题均未编造具体答案。复核记录分别在 `evals/runs/real-policy-chat-sample.codex-review.csv` 和 `evals/runs/real-policy-chat-semantic-sample.codex-review.csv`。这是一轮随机生成结果，不能把 1 题差异归因于切块；仍需修正答案核验逻辑，并经制度负责人正式复核。

含 `/chat` 的完整评测：

```bash
.venv/bin/python -m evals.run_rag_eval \
  --cases evals/runs/real-policy-cases-semantic.jsonl \
  --top-ks 1,3,5 \
  > evals/runs/real-policy-chat.json

.venv/bin/python -m evals.answer_review make \
  --report evals/runs/real-policy-chat.json \
  --output evals/runs/real-policy-chat.review.csv
```

评审人阅读 PDF 和回答后，在 CSV 中用 `0` 或 `1` 填写 `answer_correct_0_or_1`、`grounded_0_or_1`、`citation_correct_0_or_1`；对于 5 个无答案题，填写 `abstained_0_or_1`。空白表示未审，不计入分母。计算人工标注分数及覆盖量：

```bash
.venv/bin/python -m evals.answer_review score \
  --review-csv evals/runs/real-policy-chat.review.csv
```

## 指标解释

- `hit@k`：有答案题在前 k 个结果中至少一个相关 chunk 命中。
- `mrr`：有答案题首个相关 chunk 的平均倒数排名。
- `precision@k` / `recall@k`：已标注相关 chunk 在前 k 个结果中的比例／被找出的比例。一个问题只有一两个相关块时，`precision@5` 的最大值本来就很低，不应单看它判断质量。
- `evidence_coverage@k`：需要的证据组中，前 k 个结果至少命中一个等价来源的组所占比例。
- `all_evidence@k`：一道题的所有必需证据组均已在前 k 个结果中出现；用于多事实、跨表格问题。
- `keyword_match_rate`、`citation_hit_rate`：只作自动代理指标，不等同于语义正确性或引用有效性。
- 人工 `answer_accuracy`、`grounded_rate`、`citation_correct_rate`、`negative_abstention_rate`：按评审 CSV 的明确标签计算，并报告各自评审题数。
- `retrieval_latency_avg_ms` / `p95_ms`：检索接口延迟，包含嵌入和重排 API 耗时。

负例不计入检索 hit、MRR、recall；检索器仍可能返回相近文档，是否正确拒答应在 `/chat` 输出中评审。报告要同时注明数据集版本、知识库快照、模型和配置、问题数、人工复核覆盖率。当前只含一份文件，不能推断多文档检索、版本冲突、其他业务制度或生产流量的效果。

## 证据驱动流程的阶段检查

`run_rag_eval` 默认评测检索和问答，逐题记录包含 `rag_evidence` 和 `rag_stages`。读取顺序为 `retrieval` → 可选 `supplemental_retrieval` → `evidence` → `draft` → `verification` → 可选 `correction` 和再次 `verification` → `final`。多任务分别读取 `task_results`，避免混用另一任务的证据。

- 检索：核对必需的证据组是否进入候选片段，仍使用 Top K 指标。
- 整理：核对 `requirements` 的覆盖情况、用户条件是否保留、制度数值是否选自正确行列。程序验证引文存在，语义是否正确仍需核对。
- 草稿：每个结论都有事实 ID，计算结果由程序生成，引用编号由事实来源生成。
- 核验及校正：核对每条结论的状态、拒绝原因和原始问题；区分正确校正、错误拒绝与错误放行。
- 最终：分别记录结论正确性、引用正确性、问题完整回答率及资料不足时的适当拒答；不能把 `verification.passed` 当成人工正确率，也不能把 `question_answered=false` 一概当成失败。

该流程默认至少调用模型三次（整理、草稿、核验）。评测请求超时默认为 `--chat-timeout 360`，报告记录 `chat_latency_ms` 与 `usage`；复测需重启应用以加载新流程，但只改变问答流程时无需重建向量索引。保存修改前后的报告，在相同知识库、题集和服务商配置下比较，记录延迟与 token 用量。模型自核验与少量开发题的改善不能替代冻结测试集的人工复核。

## 后续扩展测试集

继续加入正式生效的差旅制度、报销细则、FAQ 及带版本日期的旧制度。让制度负责人确认标准答案、适用对象、例外和原文证据；将开发调参题与冻结测试题分开。对每个主题加入直问、改写、数值边界、跨条款计算、冲突版本和资料未提供的提问。自动生成只用于候选题，不能自动批准：

```bash
.venv/bin/python -m evals.build_rag_cases \
  --output evals/rag_cases_generated.jsonl \
  --questions-per-chunk 2

# 逐行填写 review CSV 的 reviewer_decision：keep / fix / drop。
.venv/bin/python -m evals.apply_rag_review \
  --cases evals/rag_cases_generated.jsonl \
  --review-csv evals/rag_cases_generated.review.csv \
  --output evals/rag_cases_curated.jsonl
```

空白 `reviewer_decision` 不会进入正式题集；`evidence_quote_valid` 仅检查引文是否在源 chunk 中，不能代替制度负责人核对答案。

## 演示集

`demo_corpus.jsonl` 和 `demo_cases.jsonl` 全是虚构企业数据，仅用于验证评测脚本。`run_demo_eval` 要求知识库为空，会暂时插入演示文档并清理；当前已入库真实制度，不要在当前知识库上运行演示脚本。演示分数不得作为业务效果。
