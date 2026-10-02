"""Core data model for MedGate.

Everything the gate reasons about is expressed as plain, serializable
dataclasses so that every decision is inspectable and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DeploymentScope:
    """The *exact* proposed use of a healthcare AI system.

    Authorization attaches to a scope, not to a model in general — capability
    is uneven across nearby tasks, so a score on one task does not transfer to
    another (the core argument of the "Open-Box Fallacy" paper).
    """

    task: str                 # e.g. "postpartum depression risk prediction"
    population: str           # e.g. "postpartum patients, 0-12 weeks"
    setting: str              # e.g. "community primary-care clinic"
    modality: str             # e.g. "EHR fields + PHQ-4 questionnaire"
    subgroup_key: str = "language"  # feature used for stratified verification


@dataclass
class Sample:
    """One validation record for a deployment scope.

    ``group`` is the value of the scope's ``subgroup_key`` for this record; it
    is what lets the gate do *stratified* scope verification (catching a
    sub-population blind spot that an overall accuracy score would hide).
    """

    features: Dict[str, Any]
    true_label: int           # 0 / 1 ground truth
    group: str = "all"


@dataclass
class Prediction:
    """Output of a healthcare AI system for one input.

    ``action`` is the safety channel: a safe system must be able to *refuse*
    or *escalate* to a human rather than always ``answer``.
    """

    label: int                # predicted class (0 / 1)
    confidence: float         # P(label == 1), in [0, 1]
    action: str = "answer"    # "answer" | "refuse" | "escalate"


@dataclass
class ComponentResult:
    """Outcome of one of the six verification components.

    ``score`` is a normalized 0..1 quality value. ``passed`` is the hard
    boolean (does it meet its floor?). ``mandatory`` marks components that the
    minimum-composition rule treats as non-negotiable.
    """

    name: str
    score: float
    passed: bool
    mandatory: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "mandatory": self.mandatory,
            "details": self.details,
        }


@dataclass
class GateDecision:
    """The gate's verdict for a (system, scope) pair."""

    decision: str             # "APPROVE" | "APPROVE_WITH_CONDITIONS" | "REJECT"
    vcs: float                # Verification Coverage Score, 0..100
    components: Dict[str, ComponentResult]
    conditions: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "vcs": round(self.vcs, 2),
            "conditions": self.conditions,
            "reasons": self.reasons,
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }


@dataclass
class License:
    """A revocable, scope-bound authorization to deploy.

    A license is a *permission*, not a certificate: it carries the exact scope
    it covers, any conditions, an expiry, and can be revoked at any time (by
    the post-market drift monitor or a human reviewer) with a recorded reason.
    """

    license_id: str
    scope: DeploymentScope
    decision: str
    vcs: float
    conditions: List[str] = field(default_factory=list)
    issued_at: str = "t0"
    expires_at: str = "t365"
    status: str = "ACTIVE"    # "ACTIVE" | "REVOKED" | "EXPIRED"
    revocation_reason: Optional[str] = None

    def revoke(self, reason: str, at: str = "t?") -> None:
        if self.status != "ACTIVE":
            return
        self.status = "REVOKED"
        self.revocation_reason = reason
        self._revoked_at = at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "license_id": self.license_id,
            "status": self.status,
            "decision": self.decision,
            "vcs": round(self.vcs, 2),
            "scope": {
                "task": self.scope.task,
                "population": self.scope.population,
                "setting": self.scope.setting,
                "modality": self.scope.modality,
            },
            "conditions": self.conditions,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "revocation_reason": self.revocation_reason,
        }