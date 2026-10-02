# MedGate — A Calibrated Verification Gate for Healthcare AI Deployment

**The problem the field has converged on but not yet implemented.**

A scan of 200 recent arXiv papers on AI in health care shows the field's center
of gravity is *capability* (LLMs, imaging, EHR, diagnosis, patient
communication). A smaller but growing cluster attacks the *deployment* problem:
**when is a healthcare AI system actually allowed to be used on real patients?**

The clearest statement of that standard is *"The Open-Box Fallacy: Why AI
Deployment Needs a Calibrated Verification Regime"* (arXiv:2605.10601, 2026),
which argues authorization should be **domain-scoped, independently checkable,
monitored after release, accountable, contestable, and revocable**, and
proposes a *Verification Coverage* reportable standard — citing a 53-point gap
between understanding a model and correcting its outputs, and only **9%** of
FDA-approved AI/ML device documents containing a prospective post-market
surveillance study.

**MedGate is a working reference implementation of that standard.** Given a
healthcare AI system and a proposed deployment scope, it produces:

1. a **Verification Coverage Score (VCS, 0–100)**,
2. a decision: `APPROVE` / `APPROVE_WITH_CONDITIONS` / `REJECT`,
3. a **revocable, scope-bound license**,
4. a **hash-chained, tamper-evident audit trail** making every decision
   contestable and independently re-verifiable.

See `docs/PROBLEM.md` for the full problem statement and `docs/REPORT.md` for
the design rationale mapped to the literature.

---

## Quick start

No dependencies beyond Python 3.9+ stdlib (pytest for tests).

```bash
# Run the end-to-end demo (full lifecycle: approve → restrict → drift → revoke)
python3 demo.py

# Gate a simulated system from the CLI
python3 cli.py --system clean     # -> APPROVE
python3 cli.py --system biased    # -> APPROVE_WITH_CONDITIONS
python3 cli.py --system unsafe    # -> REJECT

# Run the test suite (31 tests)
python3 -m pytest tests/ -q
```

## The six Verification Coverage components

| # | Component | What it checks | Mandatory |
|---|-----------|----------------|-----------|
| 1 | `scope_verification` | Accuracy on the *exact* proposed scope, with a 95% Wilson CI and **per-subgroup floors** (catches sub-population blind spots an overall score hides) | yes |
| 2 | `calibration` | Confidence is calibrated (ECE ≤ threshold) — the uncertainty a clinician sees is trustworthy | yes |
| 3 | `safety_audit` | A battery of high-risk inputs (e.g. self-harm ideation); the system must **refuse/escalate**, never answer | yes |
| 4 | `monitoring_plan` | An executable post-market surveillance plan with explicit drift/error thresholds and a revocation trigger (targets the 9% gap) | yes |
| 5 | `contestability` | The decision is hash-chained into an audit trail that a third party can re-verify | yes |
| 6 | `independent_check` | A second, separately-written verifier recomputes the key numbers and cross-checks the primary evaluator (no self-attestation) | no (strong signal) |

**Decision policy (minimum-composition rule):** any mandatory component fails
→ `REJECT`; all mandatory pass but a scope restriction is required (e.g. a weak
subgroup) → `APPROVE_WITH_CONDITIONS` with the restriction written into the
license; otherwise → `APPROVE`.

## Architecture

```
medgate/
├── models.py         # DeploymentScope, Sample, Prediction, License, GateDecision
├── metrics.py        # accuracy, Wilson CI, ECE, PSI, composition (stdlib only)
├── verification.py   # the six components (pure functions of predictions+scope)
├── gate.py           # MedGate: orchestrates components → VCS → decision → license
├── drift.py          # DriftMonitor: post-market PSI + error monitoring → revocation
├── audit.py          # AuditTrail: hash-chained, tamper-evident log
└── systems.py        # simulated postpartum-depression system (test double)
demo.py               # end-to-end lifecycle demo
cli.py                # command-line gate
tests/test_medgate.py # 31 tests: every component + policy + lifecycle
docs/
├── PROBLEM.md        # the innovation problem statement (grounded in 200 papers)
├── REPORT.md         # design rationale mapped to the literature
└── literature_map.csv# all 200 papers with theme tags
```

## Plugging in a real system

`MedGate.evaluate()` accepts anything that maps `features -> Prediction`
(either a callable or an object with `.predict`). To gate a real model, wrap it:

```python
class MyModel:
    def predict(self, features: dict) -> Prediction:
        p = my_model.predict_proba(features)[1]
        if needs_human_review(features):
            return Prediction(label=1, confidence=p, action="escalate")
        return Prediction(label=int(p >= 0.5), confidence=p, action="answer")

decision = gate.evaluate(MyModel(), scope, validation_samples, safety_cases)
```

The gate is agnostic to what produces the predictions — that is the point:
verification is a property of the *deployment*, not the model.

## Design principles

- **Deterministic & offline.** No network, no RNG, no ML dependencies. Every
  run reproduces byte-for-byte; the whole thing runs in well under a second.
- **Scope-bound.** Authorization attaches to a (task, population, setting,
  modality) tuple — a score on one task never transfers to another.
- **A license is a permission, not a certificate.** It can be revoked at any
  time by the drift monitor or a human, with the reason recorded on the chain.
- **No self-attestation.** The independent-check component re-derives the key
  numbers from raw inputs with a separate implementation.
- **Tamper-evident.** Every event (evaluation, decision, license, live batch,
  revocation) is hash-chained; `AuditTrail.verify()` re-proves the whole
  history offline.