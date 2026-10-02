"""MedGate — a calibrated verification gate for healthcare AI deployment.

Reference implementation of the "Verification Coverage" deployment standard
proposed in the recent AI-in-healthcare literature (see docs/PROBLEM.md).

MedGate answers the question the literature has converged on but not yet
implemented: *when is a healthcare AI system actually allowed to be used on
real patients, for a specific task, on a specific population, in a specific
setting — and when must that permission be taken back?*

The gate produces a Verification Coverage Score (VCS, 0-100) and a decision
(APPROVE / APPROVE_WITH_CONDITIONS / REJECT) plus a revocable, scope-bound
license, backed by a hash-chained, independently checkable audit trail.
"""

from .models import (
    DeploymentScope,
    Sample,
    Prediction,
    ComponentResult,
    GateDecision,
    License,
)
from .gate import MedGate, GateConfig
from .drift import DriftMonitor
from .audit import AuditTrail
from .systems import SimulatedDepressionRiskSystem, make_validation_set

__all__ = [
    "DeploymentScope",
    "Sample",
    "Prediction",
    "ComponentResult",
    "GateDecision",
    "License",
    "MedGate",
    "GateConfig",
    "DriftMonitor",
    "AuditTrail",
    "SimulatedDepressionRiskSystem",
    "make_validation_set",
]

__version__ = "0.1.0"