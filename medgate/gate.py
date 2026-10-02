"""MedGate — the calibrated verification gate.

``MedGate.evaluate(system, scope, samples, ...)`` runs all six Verification
Coverage components, computes a single Verification Coverage Score (VCS,
0-100), and returns a GateDecision plus a revocable License. Every event is
written to a hash-chained AuditTrail so the decision is contestable and
independently re-verifiable.

Decision policy (the minimum-composition rule):
  * any MANDATORY component failing            -> REJECT
  * all mandatory pass, but a scope restriction
    is required (e.g. a weak subgroup)         -> APPROVE_WITH_CONDITIONS
  * all mandatory pass, no restrictions        -> APPROVE

The VCS is a weighted, transparent aggregation so a reviewer can see *why* a
score is what it is — it is a reportable number, not a black box.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .audit import AuditTrail
from .drift import DriftConfig, DriftMonitor, LiveBatch
from .models import DeploymentScope, GateDecision, License, Prediction, Sample
from .verification import (
    SafetyCase,
    VerificationConfig,
    calibration,
    contestability,
    independent_check,
    monitoring_plan,
    safety_audit,
    scope_verification,
)

# Component weights for the VCS aggregation (sum to 1.0).
_WEIGHTS = {
    "scope_verification": 0.30,
    "calibration": 0.15,
    "safety_audit": 0.20,
    "monitoring_plan": 0.10,
    "contestability": 0.10,
    "independent_check": 0.15,
}


@dataclass
class GateConfig:
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    license_horizon_days: int = 365
    # If a subgroup is below floor, we can still approve *with a condition*
    # restricting that subgroup to human-in-the-loop, rather than hard-reject.
    allow_subgroup_condition: bool = True


class MedGate:
    """The deployment gate. One instance = one governance authority."""

    def __init__(self, cfg: Optional[GateConfig] = None) -> None:
        self.cfg = cfg or GateConfig()
        self.audit = AuditTrail()
        self._counter = itertools.count(1)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        system: Callable[[Dict], Prediction],
        scope: DeploymentScope,
        samples: Sequence[Sample],
        safety_cases: Sequence[SafetyCase],
        system_name: str = "unknown-system",
        at: str = "t0",
    ) -> GateDecision:
        vcfg = self.cfg.verification

        # Accept either a callable(features)->Prediction or an object with
        # .predict(features)->Prediction (a real model wrapper).
        predict_fn: Callable[[Dict], Prediction] = (
            system if callable(system) else system.predict
        )

        # Predict once; every component consumes the same predictions.
        preds = [predict_fn(s.features) for s in samples]

        # 1-4: the core technical components.
        comps = {
            "scope_verification": scope_verification(samples, preds, scope, vcfg),
            "calibration": calibration(samples, preds, vcfg),
            "safety_audit": safety_audit(predict_fn, safety_cases, vcfg),
            "monitoring_plan": monitoring_plan(samples, scope, vcfg),
        }

        # 5: record the (pre-contestability) decision so the audit entry
        # captures what was decided, then verify the chain.
        provisional = self._decide(comps, samples, preds)
        decision_ref = {
            "decision": provisional.decision,
            "system": system_name,
            "scope": scope.task,
            "vcs": provisional.vcs,
        }
        self.audit.append("gate.evaluate", decision_ref, at=at)
        comps["contestability"] = contestability(self.audit, decision_ref)

        # 6: independent re-check against the primary numbers.
        comps["independent_check"] = independent_check(samples, preds, comps, vcfg)

        # Final decision now includes contestability + independent check.
        decision = self._decide(comps, samples, preds, system_name=system_name)
        self.audit.append(
            "gate.decision",
            {
                "decision": decision.decision,
                "vcs": round(decision.vcs, 2),
                "system": system_name,
                "scope": scope.task,
                "conditions": decision.conditions,
                "components": {k: v.to_dict() for k, v in comps.items()},
            },
            at=at,
        )
        return decision

    # ------------------------------------------------------------------
    def issue_license(self, decision: GateDecision, scope: DeploymentScope, at: str = "t0") -> License:
        if decision.decision == "REJECT":
            # No license is issued for a rejected system; record the denial.
            self.audit.append(
                "license.denied",
                {"decision": decision.decision, "scope": scope.task,
                 "vcs": round(decision.vcs, 2), "reasons": decision.reasons},
                at=at,
            )
            raise PermissionError(
                f"Cannot issue license: decision is REJECT (VCS={decision.vcs:.1f}). "
                f"Reasons: {decision.reasons}"
            )
        lic = License(
            license_id=f"LG-{next(self._counter):05d}",
            scope=scope,
            decision=decision.decision,
            vcs=decision.vcs,
            conditions=list(decision.conditions),
            issued_at=at,
            expires_at=f"t{self.cfg.license_horizon_days}",
        )
        self.audit.append("license.issued", lic.to_dict(), at=at)
        return lic

    # ------------------------------------------------------------------
    def build_drift_monitor(self, license_obj: License, samples: Sequence[Sample]) -> DriftMonitor:
        """Instantiate the post-market monitor from the license's monitoring plan.

        Every live batch (and any revocation) is hash-chained into the audit
        trail, so post-market actions are contestable like the original
        decision.
        """
        from .metrics import composition

        vcfg = self.cfg.verification
        baseline = composition([s.group for s in samples])
        cfg = DriftConfig(
            psi_threshold=vcfg.drift_psi_threshold,
            error_excess_threshold=vcfg.drift_error_threshold,
            baseline_error_rate=0.10,
        )

        def _on_event(report: Dict) -> None:
            self.audit.append(
                "monitor.batch" if report["action"] != "REVOKED" else "license.revoked",
                report,
                at=report.get("at", "t?"),
            )

        mon = DriftMonitor(cfg, baseline, on_event=_on_event)
        self.audit.append(
            "monitor.armed",
            {"license_id": license_obj.license_id, "baseline": {k: round(v, 4) for k, v in baseline.items()}},
        )
        return mon

    # ------------------------------------------------------------------
    def _decide(
        self,
        comps: Dict[str, object],
        samples: Sequence[Sample],
        preds: Sequence[Prediction],
        system_name: str = "unknown-system",
    ) -> GateDecision:
        vcfg = self.cfg.verification
        conditions: List[str] = []
        reasons: List[str] = []

        # VCS aggregation over the components present so far.
        vcs = 0.0
        for name, cr in comps.items():
            if cr is None:
                continue
            w = _WEIGHTS.get(name, 0.0)
            vcs += w * max(0.0, min(1.0, cr.score))
        vcs *= 100.0

        mandatory_fail = [
            name for name, cr in comps.items()
            if cr is not None and cr.mandatory and not cr.passed
        ]

        # Scope restrictions: a weak subgroup becomes a condition, not a
        # hard reject, when the gate is configured to allow it.
        sv = comps.get("scope_verification")
        weak = []
        if sv is not None and not sv.passed:
            weak = sv.details.get("weak_subgroups", {})
            if self.cfg.allow_subgroup_condition and weak and not (
                sv.details["accuracy"] < vcfg.min_accuracy
                or sv.details["ci95"][0] < vcfg.min_ci_lower
            ):
                for g, a in weak.items():
                    conditions.append(
                        f"Subgroup '{g}' accuracy {a:.2f} < {vcfg.min_subgroup_accuracy:.2f}: "
                        f"restrict to human-in-the-loop review until re-verified."
                    )
                # A condition means the scope component is 'satisfied under
                # condition', so it is not counted as a hard mandatory fail.
                mandatory_fail = [m for m in mandatory_fail if m != "scope_verification"]
                reasons.append(
                    f"Restricted approval: weak subgroup(s) {list(weak)} require "
                    f"human-in-the-loop; scope-bound license issued."
                )

        if mandatory_fail:
            decision = "REJECT"
            for m in mandatory_fail:
                cr = comps[m]
                reasons.append(f"mandatory component '{m}' failed: {cr.details}")
        elif conditions:
            decision = "APPROVE_WITH_CONDITIONS"
        else:
            decision = "APPROVE"

        return GateDecision(
            decision=decision,
            vcs=vcs,
            components={k: v for k, v in comps.items() if v is not None},
            conditions=conditions,
            reasons=reasons,
        )