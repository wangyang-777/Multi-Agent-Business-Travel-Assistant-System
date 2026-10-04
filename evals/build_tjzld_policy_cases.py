"""Build reviewed candidate questions for the supplied Tianjin Green Energy travel policy.

The source PDF stays local and is deliberately excluded from Git. This script validates
every evidence phrase against the parsed PDF and binds cases to the actual chunk IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from app.config import settings
from app.etl.pipeline import (
    chunk_document,
    enforce_document_chunk_token_limit,
    parse_plain_text,
    stable_chunk_id,
)
from app.services.document_loader import load_document_from_file


SOURCE_SHA256 = "8ee3b4ac725bef40d1d7ba2655f46cfec63bc126876c543a33b66f80fb43951c"
PARENT_DOC_ID = "tjzld-travel-policy-2024"


def _source(page: int, block: str, quote: str) -> dict:
    return {"page": page, "block": block, "quote": quote}


def _either(*sources: dict) -> tuple[dict, ...]:
    """Equivalent source chunks: one of them is enough to support the fact."""
    return sources


def _case(
    case_id: str,
    question: str,
    expected_answer: str,
    case_type: str,
    *sources: dict,
    keywords: tuple[str, ...] = (),
) -> dict:
    return {
        "id": case_id,
        "question": question,
        "expected_answer": expected_answer,
        "case_type": case_type,
        "answerable": bool(sources),
        "sources": list(sources),
        "keywords": list(keywords),
    }


CASES = [
    _case("policy-01", "这份办法涵盖哪四类差旅费？", "国（境）内交通费、住宿费、伙食补助费和市内交通费。", "scope", _source(1, "paragraph", "差旅费范围包括国（境）内交通费、住宿费、伙食补助费和市内交通费")),
    _case("policy-02", "所属控股单位是否适用这份差旅费办法？", "适用，公司本部以及所属各级全资、控股单位均适用。", "scope", _source(1, "paragraph", "公司本部以及所属各级全资、控股单位")),
    _case("policy-03", "公司本部由哪个部门负责差旅费报销、审核和核算？", "财务资产部。", "responsibility", _source(1, "paragraph", "财务资产部是差旅费归口管理部门")),
    _case("policy-04", "公司本部的协议酒店库由哪个部门下发和维护？", "综合管理部。", "responsibility", _source(1, "paragraph", "负责协议酒店库的下发和维护")),
    _case("policy-05", "L1级领导因公乘飞机可按什么舱位标准？", "公务舱。", "transport_table", _either(_source(2, "table", "L1 级领导人员 | 火车软席"), _source(2, "paragraph", "二等舱  公务舱  凭据报销"))),
    _case("policy-06", "L2级领导乘飞机的规定舱位是什么？", "经济舱。", "transport_table", _either(_source(2, "table", "L2 级领导人员 | 火车软席"), _source(2, "paragraph", "L2 级领导人员  高铁/动车一等座"))),
    _case("policy-07", "其他人员乘高铁应选择几等座？", "二等座。", "transport_table", _either(_source(2, "table", "其他人员 | 火车硬席"), _source(2, "paragraph", "高铁/动车二等座"))),
    _case("policy-08", "L3级领导乘高铁可按几等座报销？", "一等座。", "transport_table", _either(_source(2, "table", "L3 级、L4 级领导人 员、外部董事"), _source(2, "paragraph", "L3 级、L4 级领导人"))),
    _case("policy-09", "L1级领导出差时，随行一人能否也乘飞机公务舱？", "不能。因工作需要随行一人可乘坐同等级交通工具，但飞机公务舱除外。", "exception", _source(2, "paragraph", "随行一人可乘坐除飞机公务舱以外的同等级交通工具")),
    _case("policy-10", "紧急公务确需乘坐超本级标准交通工具，应办什么手续？", "需提供书面说明及证明，经单位主要负责人审批；报销时附本人签字的订票时点截图。", "exception", _source(3, "paragraph", "确需乘坐超过本级标准交通工具的，需提供相关书面说明及证明")),
    _case("policy-11", "其他人员夜间坐火车超过6小时，能否直接报销软卧？", "不能直接报销。满足晚8时至次日晨7时期间乘车6小时以上等条件，并经单位主要负责人审批，方可乘坐全列软席列车软卧据实报销。", "boundary", _source(3, "paragraph", "其他人员在晚8 时至次日晨7 时期间乘车时间6 小时以上的")),
    _case("policy-12", "公司已经统一购买交通意外保险，员工还能重复购买报销吗？", "不能重复购买。", "exception", _source(3, "paragraph", "公司统一购买交通意外保险的，不再重复购买")),
    _case("policy-13", "报销往返机场的网约车费，需要什么额外凭证？", "需提供记载行程、车型和乘车时间的行程单；车型限快车及以下等级经济型车辆。", "reimbursement", _source(3, "paragraph", "网约车报销还需提供记载行程、车型和乘车时间的行程单")),
    _case("policy-14", "机场打车费用凭据报销后，市内交通补贴是否仍全额发放？", "不全额发放；往返机场的出租车或网约车视同提供一次交通工具，报销时相应扣减市内交通补贴。", "exception", _source(3, "paragraph", "报销时相应扣减出差人员市内交通补贴")),
    _case("policy-15", "出差住宿应优先选择哪类酒店？", "优先选择系统内协议酒店；无法满足住宿要求时可选系统外协议酒店。", "hotel", _source(3, "paragraph", "优先选择系统内协议酒店")),
    _case("policy-16", "因工作需要入住非协议酒店，可以原则上选择五星级酒店吗？", "原则上不得入住五星级酒店。", "hotel", _source(3, "paragraph", "原则上不得入住五星级酒店")),
    _case("policy-17", "普通员工因工作需要住北京非协议酒店，每天住宿费限额是多少？", "600元/天。", "hotel_table", _either(_source(3, "table", "北上广深 | 900 元/天 | 800 元/天 | 700 元/天 | 600 元/天"), _source(3, "paragraph", "北上广深       900 元/天        800 元/天        700 元/天     600 元/天")), keywords=("600",)),
    _case("policy-18", "L3级领导因工作需要住上海非协议酒店，每天限额是多少？", "700元/天。", "hotel_table", _either(_source(3, "table", "北上广深 | 900 元/天 | 800 元/天 | 700 元/天 | 600 元/天"), _source(3, "paragraph", "北上广深       900 元/天        800 元/天        700 元/天     600 元/天")), keywords=("700",)),
    _case("policy-19", "普通员工因工作需要住成都非协议酒店，每天限额是多少？", "成都按其他城市标准，500元/天。", "hotel_table", _either(_source(3, "table", "其他城市 | 900 元/天 | 800 元/天 | 600 元/天 | 500 元/天"), _source(3, "paragraph", "其他城市       900 元/天        800 元/天        600 元/天     500 元/天")), keywords=("500",)),
    _case("policy-20", "L2级领导住非协议酒店，其他城市与北上广深的限额有差别吗？", "没有，两个类别均为800元/天。", "hotel_table", _either(_source(3, "table", "北上广深 | 900 元/天 | 800 元/天"), _source(3, "paragraph", "北上广深       900 元/天        800 元/天")), _either(_source(3, "table", "其他城市 | 900 元/天 | 800 元/天"), _source(3, "paragraph", "其他城市       900 元/天        800 元/天")), keywords=("800",)),
    _case("policy-21", "普通员工8月去青岛住非协议酒店，住宿限额旺季最高可上浮到多少？", "青岛8月属于附件旺季，其他城市普通人员基本限额500元/天，最高上浮20%后为600元/天；上浮是最高限度，并非自动执行。", "multi_source_calculation", _either(_source(3, "table", "其他城市 | 900 元/天 | 800 元/天 | 600 元/天 | 500 元/天"), _source(3, "paragraph", "其他城市       900 元/天        800 元/天        600 元/天     500 元/天")), _source(3, "paragraph", "住宿费限额标准在旺季最高上浮20%"), _either(_source(7, "table", "青岛 | 全市 | 7-9 月"), _source(7, "paragraph", "青岛           全市                      7-9 月")), keywords=("600",)),
    _case("policy-22", "一般出差每天伙食补助标准是多少？", "每人每天100元，按自然日计算；青海、新疆、西藏另有120元标准。", "allowance", _source(4, "paragraph", "每天100 元，到青海、新疆、西藏每人每天120 元"), keywords=("100",)),
    _case("policy-23", "到新疆出差每天的伙食补助是多少？", "每人每天120元，按自然日计算。", "allowance", _source(4, "paragraph", "到青海、新疆、西藏每人每天120 元"), keywords=("120",)),
    _case("policy-24", "到青海出差3个自然日，未由其他单位供餐时，伙食补助合计多少？", "3×120元＝360元。", "calculation", _source(4, "paragraph", "到青海、新疆、西藏每人每天120 元"), keywords=("360",)),
    _case("policy-25", "出差市内交通费按自然日每天补助多少？", "每人每天80元。", "allowance", _source(4, "paragraph", "市内交通费按出差自然（日历）天数实行定额包干"), keywords=("80",)),
    _case("policy-26", "接待单位提供餐饮和交通后，补助还按原额发吗？", "不按原额发放；伙食补助费和市内交通费须按提供的餐次及交通工具情况按标准比例扣除。", "exception", _source(4, "paragraph", "按照标准比例扣除")),
    _case("policy-27", "会议统一安排食宿，会议期间还发伙食补助和市内交通费吗？", "不再发放伙食补助费和市内交通费；住宿费、伙食补助费依会议通知标准凭据报销。", "meeting", _source(4, "paragraph", "不再发放伙食补助费和市内交通费")),
    _case("policy-28", "会议未统一安排食宿时，会议期间三项费用按什么标准报销？", "住宿费、伙食补助费和市内交通费按差旅费规定报销。", "meeting", _source(4, "paragraph", "未统一安排食宿的，会议期间住宿费、伙食补助费、市内交通费按照差旅费规定报销")),
    _case("policy-29", "异地培训由培训单位统一安排食宿，第20天市内交通费上限多少？", "第20天属于超过15天、未超过30天的部分，每人每天不超过50元；统一安排食宿时不再发放伙食补助。", "training_boundary", _source(4, "paragraph", "超过15 天的，超过天数按50 元控制"), keywords=("50",)),
    _case("policy-30", "异地培训未统一安排食宿，第31天伙食和市内交通费分别控制在多少？", "第31天分别不超过70元和40元。", "training_boundary", _source(4, "paragraph", "超过30 天的，超过天数分别按70 元、40 元控制"), keywords=("70", "40")),
    _case("policy-31", "异地挂职期间所在单位统一安排就餐和住宿，还能领取哪些住宿、伙食、市内交通补助？", "工作期间不再发放伙食补助，也不再报销住宿费和市内交通费；在途期间按出差规定执行。", "secondment", _source(5, "paragraph", "所在单位统一安排就餐的，不再发放伙食补助")),
    _case("policy-32", "员工调动工作产生的差旅费由调出还是调入单位报销？", "由调入单位按差旅规定一次性报销。", "transfer", _source(5, "paragraph", "由调入单位按照差旅费规定予以一次性报销")),
    _case("policy-33", "工作调动托运行李，距离250公里时每人的凭据报销上限是多少？", "每人每公里1元以内，250公里对应每人最多250元，且须凭据报销。", "calculation", _source(5, "paragraph", "托运费，在每人每公里1 元以内凭据报销"), keywords=("250",)),
    _case("policy-34", "随同调动的16周岁子女是否属于本办法列明的家属范围？", "不属于列明的未满16周岁子女范围；原文要求未满16周岁。", "boundary", _source(5, "paragraph", "未满16 周岁的子女"), keywords=("16",)),
    _case("policy-35", "出差时事先获批绕道回家，绕道和在家期间的住宿、伙食和市内交通费可报销吗？", "不可报销；绕道交通费超出出差直线单程交通费的部分由个人自理。", "exception", _source(5, "paragraph", "绕道和在家期间不予报销住宿费、伙食补助和市内交通费")),
    _case("policy-36", "出差行程改变时，原来的一事一批审批是否足够？", "不足够；出差须事前一事一批审批并注明行程和事由，改变行程需另行审批。", "approval", _source(5, "paragraph", "改变行程需另行审批")),
    _case("policy-37", "原则上，差旅费用增幅可以高于利润总额增幅吗？", "原则上不得高于利润总额增幅。", "budget", _source(5, "paragraph", "差旅费用增幅不得高于利润总额增幅")),
    _case("policy-38", "原2022年的天津广宇发展股份有限公司差旅费管理办法是否仍同时有效？", "不同时有效；本办法生效时，津广宇财〔2022〕117号同时废止。", "version", _source(6, "paragraph", "津广宇财〔2022〕117 号）同时废止")),
    _case("policy-39", "三亚市11月是否在附件列出的住宿旺季期间？", "是。三亚市旺季期间为10月至次年4月。", "seasonal_table", _either(_source(7, "table", "三亚市 | 10-4 月"), _source(7, "paragraph", "三亚市                     10-4 月"))),
    _case("policy-40", "张家口市1月是否属于附件列出的住宿旺季？", "是。张家口市旺季包括11月至次年3月。", "seasonal_table", _either(_source(7, "table", "张家口市 | 7-9 月、11-3 月"), _source(7, "paragraph", "张家口市                7-9 月、11-3 月"))),
    _case("policy-41", "这份办法写明董事会审议通过的具体日期是哪一天？", "文档未载明具体董事会通过日期，需查董事会决议或正式发文。", "unanswerable"),
    _case("policy-42", "依据这份办法，赴美国出差的住宿费每天上限是多少美元？", "本办法未规定境外住宿美元限额，不能据此给出数值。", "unanswerable"),
    _case("policy-43", "接待单位提供早餐时，伙食补助具体扣除百分之多少？", "本办法仅说按标准比例扣除，未载明早餐的具体扣除百分比。", "unanswerable"),
    _case("policy-44", "这份办法列出的上海协议酒店名称有哪些？", "文档未列出协议酒店名单，需查看综合管理部维护的酒店库。", "unanswerable"),
    _case("policy-45", "机场出租车已凭据报销时，市内交通补贴具体扣减多少元？", "本办法要求相应扣减，但未写明精确扣减金额。", "unanswerable"),
]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def build_cases(pdf_path: Path) -> list[dict]:
    data = pdf_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"PDF SHA256 mismatch: expected {SOURCE_SHA256}, got {digest}")
    loaded = load_document_from_file(pdf_path.name, data)
    chunks = enforce_document_chunk_token_limit(
        chunk_document(
            parse_plain_text(loaded.text),
            source_format=".pdf",
            max_chars=3000,
            overlap_ratio=0.15,
            source_metadata=loaded.metadata,
        ),
        max_tokens=settings.embedding_chunk_max_tokens,
    )
    indexed = [
        (stable_chunk_id(PARENT_DOC_ID, index, chunk.content), chunk)
        for index, chunk in enumerate(chunks)
    ]
    output: list[dict] = []
    for spec in CASES:
        relevant_ids: list[str] = []
        relevance_groups: list[list[str]] = []
        evidence: list[dict] = []
        for group_index, group in enumerate(spec["sources"], start=1):
            alternatives = (group,) if isinstance(group, dict) else group
            group_ids: list[str] = []
            for source in alternatives:
                matches = [
                    doc_id
                    for doc_id, chunk in indexed
                    if chunk.metadata.get("page_number") == source["page"]
                    and source["block"] in chunk.metadata.get("block_types", [])
                    and _compact(source["quote"]) in _compact(chunk.content)
                ]
                if len(matches) != 1:
                    raise ValueError(f"{spec['id']}: evidence must match exactly one source chunk: {source}; matches={matches}")
                if matches[0] not in group_ids:
                    group_ids.append(matches[0])
                if matches[0] not in relevant_ids:
                    relevant_ids.append(matches[0])
                evidence.append({**source, "chunk_id": matches[0], "group": group_index})
            relevance_groups.append(group_ids)
        output.append({
            **{key: value for key, value in spec.items() if key != "sources"},
            "relevant_ids": relevant_ids,
            "relevance_groups": relevance_groups,
            "source_pdf_sha256": digest,
            "source_parent_doc_id": PARENT_DOC_ID,
            "evidence": evidence,
            "review": {"status": "pending_policy_owner", "notes": "确认制度生效版本及标准答案"},
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="evals/source_documents/1219916518.pdf")
    parser.add_argument("--output", default="evals/real_policy_cases.jsonl")
    args = parser.parse_args()
    rows = build_cases(Path(args.pdf))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    review_path = Path("evals/runs/real-policy-case-review.csv")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "reviewer_decision", "corrected_question", "corrected_answer",
        "corrected_relevant_ids", "notes", "case_type", "question", "expected_answer",
        "relevant_ids", "evidence",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "case_type": row["case_type"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "relevant_ids": json.dumps(row["relevant_ids"], ensure_ascii=False),
                "evidence": json.dumps(row["evidence"], ensure_ascii=False),
            })
    print(json.dumps({"cases": len(rows), "answerable": sum(row["answerable"] for row in rows), "output": str(output), "review_csv": str(review_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
