"""Document-independent evidence records and deterministic validation for RAG."""

from __future__ import annotations

import ast
import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,127}$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
NumericValue = Annotated[str, StringConstraints(min_length=1, max_length=64)]
Unit = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class EvidenceError(ValueError):
    """The model output cannot be linked to the supplied question or sources."""

    def __init__(
        self, message: str, *, code: str = "evidence_validation_failed",
        field: str | None = None, record_id: str | None = None,
        reference_ids: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.record_id = record_id
        self.reference_ids = reference_ids or []
        self.schema_errors: list[dict[str, Any]] = []
        self.invalid_output: str | None = None
        self.usage: dict[str, int] | None = None

    def diagnostic(self) -> dict[str, Any]:
        # No source text, model response, or arbitrary exception message is exposed.
        result: dict[str, Any] = {"code": self.code}
        if self.field:
            result["field"] = self.field
        if self.record_id and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", self.record_id):
            result["record_id"] = self.record_id
        if self.reference_ids:
            result["reference_ids"] = [
                fid for fid in self.reference_ids
                if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", fid)
            ][:20]
        if self.schema_errors:
            result["fields"] = self.schema_errors
        return result


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSpan(Record):
    chunk_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    quote: Text


class UserFact(Record):
    id: Identifier
    statement: Text
    quote: Text
    value: NumericValue | None = None
    unit: Unit | None = None


class PolicyFact(Record):
    id: Identifier
    statement: Text
    sources: list[SourceSpan] = Field(min_length=1, max_length=5)
    value: NumericValue | None = None
    unit: Unit | None = None


class Requirement(Record):
    id: Identifier
    description: Text
    status: Literal["covered", "missing", "conflict"]
    fact_ids: list[Identifier] = Field(default_factory=list, max_length=20)
    reason: Text


class Calculation(Record):
    id: Identifier
    expression: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    unit: Unit | None = None
    description: Text


class EvidencePack(Record):
    user_facts: list[UserFact] = Field(default_factory=list, max_length=20)
    policy_facts: list[PolicyFact] = Field(default_factory=list, max_length=30)
    requirements: list[Requirement] = Field(min_length=1, max_length=15)
    calculations: list[Calculation] = Field(default_factory=list, max_length=10)
    search_queries: list[Text] = Field(default_factory=list, max_length=2)


class AnswerClaim(Record):
    id: Identifier
    text: Text
    kind: Literal["policy", "calculation", "conditional", "limitation"]
    fact_ids: list[Identifier] = Field(default_factory=list, max_length=20)
    requirement_ids: list[Identifier] = Field(default_factory=list, max_length=15)


class AnswerDraft(Record):
    claims: list[AnswerClaim] = Field(min_length=1, max_length=20)


class ClaimReview(Record):
    claim_id: Identifier
    status: Literal["supported", "contradicted", "insufficient"]
    reason: Text


class AnswerReview(Record):
    claims: list[ClaimReview] = Field(min_length=1, max_length=20)
    question_answered: bool
    missing_information: list[Text] = Field(default_factory=list, max_length=15)
    human_review_required: bool = False
    review_reasons: list[Text] = Field(default_factory=list, max_length=15)


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _chinese_integer(text: str) -> int:
    digits = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    if not any(char in units for char in text):
        return int("".join(str(digits[char]) for char in text))
    total = section = number = 0
    for char in text:
        if char in digits:
            number = digits[char]
        elif char == "万":
            total += (section + number or 1) * units[char]
            section = number = 0
        else:
            section += (number or 1) * units[char]
            number = 0
    return total + section + number


def numbers(text: str) -> set[Decimal]:
    text = unicodedata.normalize("NFKC", text)
    values = {Decimal(x.replace(",", "")) for x in re.findall(r"(?<!\d)-?\d+(?:,\d{3})*(?:\.\d+)?", text)}
    for match in re.findall(r"[零〇一二两三四五六七八九十百千万]+", text):
        values.add(Decimal(_chinese_integer(match)))
    return values


def _decimal(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise EvidenceError("numeric fact is not a decimal", code="invalid_numeric_value") from exc
    if not result.is_finite() or abs(result) > Decimal("1e15"):
        raise EvidenceError("numeric fact is outside the supported range", code="numeric_value_out_of_range")
    return result


def evaluate(expression: str, values: dict[str, Decimal]) -> tuple[str | bool, set[str]]:
    """Evaluate bounded arithmetic/comparisons; no eval, calls, access or powers."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise EvidenceError("invalid calculation expression", code="invalid_expression") from exc
    if len(list(ast.walk(tree))) > 50:
        raise EvidenceError("calculation is too complex", code="expression_too_complex")
    used: set[str] = set()

    def visit(node: ast.AST) -> Decimal | bool:
        if isinstance(node, ast.Name) and node.id in values:
            used.add(node.id)
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return _decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if isinstance(value, bool):
                raise EvidenceError("boolean arithmetic is not supported", code="boolean_arithmetic")
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(left, bool) or isinstance(right, bool):
                raise EvidenceError("boolean arithmetic is not supported", code="boolean_arithmetic")
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif right:
                result = left / right
            else:
                raise EvidenceError("division by zero", code="division_by_zero")
            if not result.is_finite() or abs(result) > Decimal("1e15"):
                raise EvidenceError("calculation result is outside the supported range", code="calculation_out_of_range")
            return result
        if isinstance(node, ast.Compare):
            operands = [visit(node.left), *(visit(x) for x in node.comparators)]
            tests = []
            for left, op, right in zip(operands, node.ops, operands[1:]):
                if isinstance(op, ast.Lt): tests.append(left < right)
                elif isinstance(op, ast.LtE): tests.append(left <= right)
                elif isinstance(op, ast.Gt): tests.append(left > right)
                elif isinstance(op, ast.GtE): tests.append(left >= right)
                elif isinstance(op, ast.Eq): tests.append(left == right)
                elif isinstance(op, ast.NotEq): tests.append(left != right)
                else: raise EvidenceError("unsupported comparison", code="unsupported_comparison")
            return all(tests)
        if isinstance(node, ast.Name):
            raise EvidenceError(
                "calculation references an unknown or nonnumeric fact", code="unknown_numeric_fact",
                reference_ids=[node.id],
            )
        raise EvidenceError("only fact IDs and arithmetic/comparison operators are allowed", code="unsupported_expression")

    with localcontext() as context:
        context.prec = 28
        result = visit(tree.body)
    if not used:
        raise EvidenceError("calculation must use supplied facts", code="calculation_without_facts")
    if isinstance(result, bool):
        return result, used
    return format(result, "f").rstrip("0").rstrip(".") if "." in format(result, "f") else str(result), used


def validate_pack(pack: EvidencePack, question: str, citations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Check exact provenance before any extracted fact is used in an answer."""
    sources = {str(c.get("chunk_id")): str(c.get("content") or "") for c in citations}
    records: dict[str, dict[str, Any]] = {}
    values: dict[str, Decimal] = {}
    for fact in [*pack.user_facts, *pack.policy_facts]:
        if fact.id in records:
            raise EvidenceError("fact IDs must be unique", code="duplicate_fact_id", field="id", record_id=fact.id)
        if isinstance(fact, UserFact):
            quotes, chunk_ids = [fact.quote], []
            if normalized(fact.quote) not in normalized(question):
                raise EvidenceError(f"user fact {fact.id} is not quoted from the original question", code="invalid_user_quote", field="quote", record_id=fact.id)
        else:
            quotes, chunk_ids = [], []
            for index, span in enumerate(fact.sources):
                if span.chunk_id not in sources:
                    raise EvidenceError(f"policy fact {fact.id} has an unknown source", code="unknown_source", field=f"sources.{index}.chunk_id", record_id=fact.id)
                if normalized(span.quote) not in normalized(sources[span.chunk_id]):
                    raise EvidenceError(f"policy fact {fact.id} has an invalid source quote", code="invalid_source_quote", field=f"sources.{index}.quote", record_id=fact.id)
                quotes.append(span.quote)
                chunk_ids.append(span.chunk_id)
        if fact.value is not None:
            try:
                value = _decimal(fact.value)
            except EvidenceError as exc:
                exc.field, exc.record_id = "value", fact.id
                raise
            if value not in numbers("\n".join(quotes)):
                raise EvidenceError(f"numeric value for {fact.id} is absent from its source", code="numeric_value_not_in_quote", field="value", record_id=fact.id)
            values[fact.id] = value
        records[fact.id] = {**fact.model_dump(), "chunk_ids": list(dict.fromkeys(chunk_ids))}
    for calculation in pack.calculations:
        if calculation.id in records:
            raise EvidenceError("calculation IDs must be unique", code="duplicate_calculation_id", field="id", record_id=calculation.id)
        try:
            result, used = evaluate(calculation.expression, values)
        except EvidenceError as exc:
            exc.field, exc.record_id = "expression", calculation.id
            raise
        # Policy rates and thresholds must come from facts, rather than being
        # smuggled into an otherwise valid formula as invented constants.
        constants = {
            _decimal(str(node.value)) for node in ast.walk(ast.parse(calculation.expression, mode="eval"))
            if isinstance(node, ast.Constant) and type(node.value) in (int, float)
        }
        if constants - {Decimal(0), Decimal(1), Decimal(100)}:
            raise EvidenceError("calculation constants must use source fact IDs (except 0, 1 and 100)", code="unbound_calculation_constant", field="expression", record_id=calculation.id)
        records[calculation.id] = {
            **calculation.model_dump(), "result": result, "fact_ids": sorted(used),
            "chunk_ids": list(dict.fromkeys(c for fid in sorted(used) for c in records[fid]["chunk_ids"])),
        }
    requirement_ids: set[str] = set()
    for requirement in pack.requirements:
        if requirement.id in requirement_ids:
            raise EvidenceError("invalid requirement references: duplicate ID", code="duplicate_requirement_id", field="id", record_id=requirement.id)
        unknown = sorted(set(requirement.fact_ids) - records.keys())
        if unknown:
            raise EvidenceError("invalid requirement references", code="unknown_requirement_fact", field="fact_ids", record_id=requirement.id, reference_ids=unknown)
        requirement_ids.add(requirement.id)
        if requirement.status == "covered" and not requirement.fact_ids:
            raise EvidenceError("covered requirements must reference evidence or user facts", code="covered_requirement_without_facts", field="fact_ids", record_id=requirement.id)
    return records


def validate_draft(draft: AnswerDraft, pack: EvidencePack, records: dict[str, dict[str, Any]]) -> None:
    ids: set[str] = set()
    requirements = {r.id: r for r in pack.requirements}
    for claim in draft.claims:
        if claim.id in ids or not claim.text.strip():
            raise EvidenceError("claim IDs must be unique and text must not be blank", code="invalid_claim_id_or_text", field="id" if claim.id in ids else "text", record_id=claim.id)
        ids.add(claim.id)
        unknown_facts = sorted(set(claim.fact_ids) - records.keys())
        unknown_requirements = sorted(set(claim.requirement_ids) - requirements.keys())
        if unknown_facts or unknown_requirements:
            raise EvidenceError("claim references unknown facts or requirements", code="unknown_claim_reference", field="fact_ids" if unknown_facts else "requirement_ids", record_id=claim.id, reference_ids=unknown_facts or unknown_requirements)
        if claim.kind != "limitation" and not any(records[f].get("chunk_ids") for f in claim.fact_ids):
            raise EvidenceError("policy conclusions must have policy evidence", code="claim_without_policy_evidence", field="fact_ids", record_id=claim.id)
        if claim.kind == "limitation" and not claim.requirement_ids:
            raise EvidenceError("limitations must identify the unmet requirement", code="limitation_without_requirement", field="requirement_ids", record_id=claim.id)
        if claim.kind == "calculation" and not any("result" in records[f] for f in claim.fact_ids):
            raise EvidenceError("calculation claims must reference a checked calculation", code="calculation_without_checked_result", field="fact_ids", record_id=claim.id)


def deterministic_claim_errors(claim: AnswerClaim, records: dict[str, dict[str, Any]]) -> list[str]:
    """Check calculation results without comparing unrelated table amounts."""
    errors = []
    if claim.kind == "calculation":
        numeric_results = [r["result"] for fid in claim.fact_ids if "result" in (r := records[fid]) and not isinstance(r["result"], bool)]
        if numeric_results and not all(_decimal(result) in numbers(claim.text) for result in numeric_results):
            errors.append("答案中的计算结果与程序校验的结果不一致")
    amounts = {Decimal(x.replace(",", "")) for x in re.findall(r"(?:\b(?:CNY|RMB)\s*|[¥￥]\s*)(\d+(?:,\d{3})*(?:\.\d+)?)", claim.text, re.I)}
    amounts.update(Decimal(x.replace(",", "")) for x in re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:元|CNY\b|RMB\b)", claim.text, re.I))
    allowed: set[Decimal] = set()
    visited: set[str] = set()

    def collect(fid: str) -> None:
        if fid in visited:
            return
        visited.add(fid)
        record = records[fid]
        value = record.get("result", record.get("value"))
        if value is not None and not isinstance(value, bool) and ("result" in record or record.get("unit") is None or any(u in str(record.get("unit") or "").upper() for u in ("元", "CNY", "RMB", "¥", "￥"))):
            allowed.add(_decimal(str(value)))
        elif value is None:
            for source in record.get("sources", []):
                allowed.update(numbers(source["quote"]))
        for parent in record.get("fact_ids", []):
            collect(parent)

    for fid in claim.fact_ids:
        collect(fid)
    if amounts and claim.kind != "limitation" and amounts - allowed:
        errors.append("答案中的金额不在所引用的数值事实或已校验计算结果中")
    return errors


def claim_sources(claim: AnswerClaim, records: dict[str, dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(c for fid in claim.fact_ids for c in records[fid]["chunk_ids"]))


def render_draft(draft: AnswerDraft, records: dict[str, dict[str, Any]], citations: list[dict[str, Any]]) -> str:
    indices = {str(c.get("chunk_id")): i for i, c in enumerate(citations, 1)}
    parts = []
    for claim in draft.claims:
        # Citations are assigned from validated provenance, never invented by the model.
        text = re.sub(r"\[\d+\]", "", claim.text).strip()
        refs = "".join(f"[{indices[c]}]" for c in claim_sources(claim, records))
        parts.append(text + refs)
    return "\n\n".join(parts)


def verification_result(review: AnswerReview, draft: AnswerDraft, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {r.claim_id: r for r in review.claims}
    if len(by_id) != len(review.claims) or set(by_id) != {c.id for c in draft.claims}:
        raise EvidenceError("review must cover every claim exactly once", code="review_claim_coverage", field="claims")
    mapping = []
    for claim in draft.claims:
        verdict = by_id[claim.id]
        errors = deterministic_claim_errors(claim, records)
        status = "contradicted" if errors else verdict.status
        mapping.append({
            "claim_id": claim.id, "claim": claim.text, "kind": claim.kind,
            "status": "conflict" if status == "contradicted" else "partial" if status == "insufficient" else "supported",
            "fact_ids": claim.fact_ids, "supporting_chunk_ids": claim_sources(claim, records),
            "reason": "；".join(errors) if errors else verdict.reason,
        })
    passed = all(c["status"] == "supported" for c in mapping)
    return {
        "passed": passed, "reason": "all_claims_supported" if passed else "claim_evidence_mismatch",
        "method": "evidence_semantic_review", "checked_terms": [], "unsupported_terms": [],
        "question_answered": review.question_answered,
        "missing_information": review.missing_information, "claim_evidence_map": mapping,
        "human_review_required": review.human_review_required,
        "review_reasons": review.review_reasons,
    }
