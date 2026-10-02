"""Test suite for MedGate.

Covers each of the six Verification Coverage components individually, the
end-to-end gate decision policy, license lifecycle, drift-based revocation,
and audit-trail tamper-evidence.
"""
from __future__ import annotations

import pytest

from medgate import (
    DeploymentScope,
    GateConfig,
    License,
    MedGate,
    SimulatedDepressionRiskSystem,
    make_validation_set,
)
from medgate.audit import AuditTrail
from medgate.drift import DriftConfig, DriftMonitor, LiveBatch
from medgate.metrics import (
    accuracy,
    composition,
    ece,
    population_stability_index,
    wilson_ci,
)
from medgate.models import Prediction, Sample
from medgate.systems import standard_safety_cases
from medgate.verification import (
    SafetyCase,
    VerificationConfig,
    calibration,
    independent_check,
    monitoring_plan,
    safety_audit,
    scope_verification,
)

SCOPE = DeploymentScope(
    task="postpartum depression risk prediction",
    population="postpartum patients, 0-12 weeks",
    setting="community primary-care clinic",
    modality="EHR fields + PHQ-4 questionnaire",
    subgroup_key="language",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_accuracy_basic():
    assert accuracy([(1, 1), (0, 0), (1, 0)]) == pytest.approx(2 / 3)
    assert accuracy([]) == 0.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(90, 100)
    assert 0.0 <= lo <= 0.9 <= hi <= 1.0
    # Perfect proportion gives a tight interval at the top.
    lo2, hi2 = wilson_ci(1000, 1000)
    assert lo2 > 0.99
    assert hi2 == 1.0


def test_ece_perfect_calibration():
    # Mean confidence equals empirical accuracy in the bin -> ECE = 0.
    confs = [0.5, 0.5]
    labels = [0, 1]
    assert ece(confs, labels) == pytest.approx(0.0)


def test_ece_bad_calibration():
    # High confidence but wrong -> large ECE.
    confs = [0.95, 0.95, 0.95, 0.95]
    labels = [0, 0, 0, 0]
    assert ece(confs, labels) > 0.5


def test_psi_identical_is_zero():
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_psi_shift_is_positive():
    assert population_stability_index([0.8, 0.2], [0.2, 0.8]) > 0.25


def test_composition():
    c = composition(["a", "a", "b"])
    assert c == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


# ---------------------------------------------------------------------------
# Component 1: scope verification
# ---------------------------------------------------------------------------
def test_scope_verification_passes_clean():
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    preds = [sys.predict(s.features) for s in samples]
    res = scope_verification(samples, preds, SCOPE, VerificationConfig())
    assert res.passed is True
    assert res.details["accuracy"] >= 0.85


def test_scope_verification_catches_subgroup_blind_spot():
    sys = SimulatedDepressionRiskSystem(es_bias=1.0)
    samples = make_validation_set(n=400)
    preds = [sys.predict(s.features) for s in samples]
    res = scope_verification(samples, preds, SCOPE, VerificationConfig())
    # The Spanish subgroup should be flagged as weak.
    assert "es" in res.details["weak_subgroups"]


# ---------------------------------------------------------------------------
# Component 2: calibration
# ---------------------------------------------------------------------------
def test_calibration_clean_passes():
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    preds = [sys.predict(s.features) for s in samples]
    res = calibration(samples, preds, VerificationConfig())
    assert res.passed is True
    assert res.details["ece"] <= 0.10


# ---------------------------------------------------------------------------
# Component 3: safety audit
# ---------------------------------------------------------------------------
def test_safety_audit_safe_system_passes():
    sys = SimulatedDepressionRiskSystem(unsafe_on_high_risk=False)
    res = safety_audit(sys.predict, standard_safety_cases(), VerificationConfig())
    assert res.passed is True
    assert res.details["safe"] == res.details["n_cases"]


def test_safety_audit_unsafe_system_fails():
    sys = SimulatedDepressionRiskSystem(unsafe_on_high_risk=True)
    res = safety_audit(sys.predict, standard_safety_cases(), VerificationConfig())
    assert res.passed is False
    assert res.details["safe"] == 0


# ---------------------------------------------------------------------------
# Component 4: monitoring plan
# ---------------------------------------------------------------------------
def test_monitoring_plan_complete():
    samples = make_validation_set(n=400)
    res = monitoring_plan(samples, SCOPE, VerificationConfig())
    assert res.passed is True
    assert "revocation_trigger" in res.details
    assert res.details["baseline_composition"]["es"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Component 5: contestability (audit trail)
# ---------------------------------------------------------------------------
def test_audit_chain_verifies():
    trail = AuditTrail()
    trail.append("event", {"x": 1}, at="t0")
    trail.append("event", {"y": 2}, at="t1")
    assert trail.verify() is True


def test_audit_chain_detects_tampering():
    trail = AuditTrail()
    trail.append("event", {"x": 1}, at="t0")
    trail.append("event", {"y": 2}, at="t1")
    # Tamper with a stored entry.
    trail._entries[0]["data"] = {"x": 999}
    assert trail.verify() is False


def test_contestability_component():
    trail = AuditTrail()
    trail.append("gate.decision", {"decision": "APPROVE"}, at="t0")
    from medgate.verification import contestability

    res = contestability(trail, {"decision": "APPROVE"})
    assert res.passed is True


# ---------------------------------------------------------------------------
# Component 6: independent check
# ---------------------------------------------------------------------------
def test_independent_check_agrees():
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    preds = [sys.predict(s.features) for s in samples]
    vcfg = VerificationConfig()
    primary = {
        "scope_verification": scope_verification(samples, preds, SCOPE, vcfg),
        "calibration": calibration(samples, preds, vcfg),
    }
    res = independent_check(samples, preds, primary, vcfg)
    assert res.passed is True
    assert res.details["mismatches"] == []


def test_independent_check_detects_mismatch():
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    preds = [sys.predict(s.features) for s in samples]
    vcfg = VerificationConfig()
    primary = {
        "scope_verification": scope_verification(samples, preds, SCOPE, vcfg),
        "calibration": calibration(samples, preds, vcfg),
    }
    # Corrupt the primary's reported accuracy to force a mismatch.
    primary["scope_verification"].details["accuracy"] = 0.123
    res = independent_check(samples, preds, primary, vcfg)
    assert res.passed is False
    assert "accuracy" in res.details["mismatches"]


# ---------------------------------------------------------------------------
# End-to-end gate decision policy
# ---------------------------------------------------------------------------
def test_gate_approves_clean_system():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases(), system_name=sys.name)
    assert d.decision == "APPROVE"
    assert d.vcs > 90
    assert d.conditions == []


def test_gate_restricts_biased_system():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(es_bias=1.0)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases(), system_name=sys.name)
    assert d.decision == "APPROVE_WITH_CONDITIONS"
    assert any("es" in c for c in d.conditions)


def test_gate_rejects_unsafe_system():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(unsafe_on_high_risk=True)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases(), system_name=sys.name)
    assert d.decision == "REJECT"
    assert any("safety_audit" in r for r in d.reasons)


def test_gate_rejects_unverifiable_scope():
    # A system that is just wrong everywhere should be rejected.
    class BadSystem:
        def predict(self, features):
            return Prediction(label=0, confidence=0.5, action="answer")

    gate = MedGate()
    samples = make_validation_set(n=400)
    d = gate.evaluate(BadSystem(), SCOPE, samples, standard_safety_cases(),
                      system_name="bad")
    assert d.decision == "REJECT"


# ---------------------------------------------------------------------------
# License lifecycle
# ---------------------------------------------------------------------------
def test_license_issued_for_approval():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases())
    lic = gate.issue_license(d, SCOPE, at="t100")
    assert lic.status == "ACTIVE"
    assert lic.decision == "APPROVE"


def test_license_denied_for_rejection():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(unsafe_on_high_risk=True)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases())
    with pytest.raises(PermissionError):
        gate.issue_license(d, SCOPE, at="t100")


def test_license_revocation_is_recorded():
    lic = License(
        license_id="LG-X",
        scope=SCOPE,
        decision="APPROVE",
        vcs=95.0,
    )
    assert lic.status == "ACTIVE"
    lic.revoke("drift detected", at="t200")
    assert lic.status == "REVOKED"
    assert lic.revocation_reason == "drift detected"
    # Revoking an already-revoked license is a no-op.
    lic.revoke("again", at="t300")
    assert lic.revocation_reason == "drift detected"


# ---------------------------------------------------------------------------
# Drift monitor -> automatic revocation
# ---------------------------------------------------------------------------
def test_drift_monitor_revokes_on_population_shift():
    lic = License(license_id="LG-D", scope=SCOPE, decision="APPROVE", vcs=95.0)
    mon = DriftMonitor(
        DriftConfig(psi_threshold=0.25, error_excess_threshold=0.15, baseline_error_rate=0.10),
        {"en": 0.8, "es": 0.2},
    )
    # On-distribution batch: no revocation.
    r1 = mon.observe(LiveBatch(["en"] * 300 + ["es"] * 80, errors=40, n=380), lic, at="t1")
    assert r1["action"] == "monitor"
    assert lic.status == "ACTIVE"
    # Heavy shift: revocation.
    r2 = mon.observe(LiveBatch(["en"] * 100 + ["es"] * 300, errors=80, n=400), lic, at="t2")
    assert r2["action"] == "REVOKED"
    assert lic.status == "REVOKED"


def test_drift_monitor_revokes_on_error_breach():
    lic = License(license_id="LG-E", scope=SCOPE, decision="APPROVE", vcs=95.0)
    mon = DriftMonitor(
        DriftConfig(psi_threshold=0.25, error_excess_threshold=0.15, baseline_error_rate=0.10),
        {"en": 0.8, "es": 0.2},
    )
    # Same distribution, but error rate far above baseline.
    r = mon.observe(LiveBatch(["en"] * 300 + ["es"] * 80, errors=100, n=380), lic, at="t1")
    assert r["action"] == "REVOKED"
    assert lic.status == "REVOKED"


def test_drift_monitor_emits_events():
    events = []
    lic = License(license_id="LG-F", scope=SCOPE, decision="APPROVE", vcs=95.0)
    mon = DriftMonitor(
        DriftConfig(psi_threshold=0.25, error_excess_threshold=0.15, baseline_error_rate=0.10),
        {"en": 0.8, "es": 0.2},
        on_event=lambda r: events.append(r),
    )
    mon.observe(LiveBatch(["en"] * 300 + ["es"] * 80, errors=40, n=380), lic, at="t1")
    assert len(events) == 1
    assert events[0]["license_id"] == "LG-F"


# ---------------------------------------------------------------------------
# Full lifecycle integration (mirrors demo.py)
# ---------------------------------------------------------------------------
def test_full_lifecycle_clean_approve():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases())
    lic = gate.issue_license(d, SCOPE, at="t100")
    assert lic.status == "ACTIVE"
    assert gate.audit.verify() is True


def test_full_lifecycle_biased_restricted_then_revoked():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(es_bias=1.0)
    samples = make_validation_set(n=400)
    d = gate.evaluate(sys, SCOPE, samples, standard_safety_cases())
    assert d.decision == "APPROVE_WITH_CONDITIONS"
    lic = gate.issue_license(d, SCOPE, at="t200")
    mon = gate.build_drift_monitor(lic, samples)
    mon.observe(LiveBatch(["en"] * 300 + ["es"] * 80, errors=40, n=380), lic, at="t210")
    assert lic.status == "ACTIVE"
    mon.observe(LiveBatch(["en"] * 100 + ["es"] * 300, errors=80, n=400), lic, at="t220")
    assert lic.status == "REVOKED"
    # The revocation must be in the audit trail and the chain must verify.
    assert gate.audit.verify() is True
    events = [e["event"] for e in gate.audit.entries()]
    assert "license.revoked" in events


def test_audit_tamper_breaks_chain_after_lifecycle():
    gate = MedGate()
    sys = SimulatedDepressionRiskSystem(es_bias=0.0)
    samples = make_validation_set(n=400)
    gate.evaluate(sys, SCOPE, samples, standard_safety_cases())
    assert gate.audit.verify() is True
    # Tamper with an entry.
    gate.audit._entries[0]["data"]["vcs"] = 0.0
    assert gate.audit.verify() is False