"""Public request and response models for the Aegis gateway."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DecisionVerdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


class EvaluationPhase(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ActionContext(BaseModel):
    """Raw context captured from a tool invocation."""

    agent_name: str
    tool_name: str
    tool_args: dict = Field(default_factory=dict)
    user_id: str | None = None
    session_id: str | None = None
    has_human_approval: bool = False
    timestamp: datetime | None = None

    def model_post_init(self, _context: object) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


class ClassifiedAction(BaseModel):
    """Action after LLM classification against the governance ontology."""

    context: ActionContext
    action_types: list[str] = Field(
        description="Ontology IRIs, e.g. 'aegis:FinancialAction'",
    )
    risk_level: str = Field(
        description="Risk-level IRI, e.g. 'aegis:HighRisk'",
    )
    involves_amount: float | None = None
    target_entity: str | None = None
    reasoning: str = ""


class ResultClassification(BaseModel):
    """LLM classification of tool result content."""

    contains_pii: bool = False
    contains_phi: bool = False
    contains_credentials: bool = False
    data_categories: list[str] = Field(default_factory=list)
    risk_assessment: str = ""
    reasoning: str = ""


class GovernanceDecision(BaseModel):
    """Final governance decision produced by the SHACL validator."""

    action: ClassifiedAction
    allowed: bool
    violations: list[str] = Field(default_factory=list)
    violated_shapes: list[str] = Field(default_factory=list)
    decision: DecisionVerdict = DecisionVerdict.ALLOW
    evaluation_phase: EvaluationPhase = EvaluationPhase.INPUT
    action_id: str | None = None
    result_classification: ResultClassification | None = None
    classify_ms: float | None = None
    validate_ms: float | None = None
    total_ms: float | None = None
