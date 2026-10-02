# MedGate — Design Report

How each design decision maps to the evidence in the 200-paper corpus.

## 1. Why a *gate* and not a *benchmark*

**Evidence.** The corpus is dominated by capability work (97 patient-communication
papers, 65 diagnosis/prediction, 47 LLM papers). Each deployment paper
(arXiv:2603.17234 surgical triage; the cancer-summary evaluation; the maternal
chatbot) builds its own bespoke evaluation. The Open-Box Fallacy paper
(arXiv:2605.10601) names the failure mode: capability scores are
*model-level*, but deployment authority must be *use-level* — "model capability
is uneven across nearby tasks, so authorization must attach to a specific use
rather than to a model in general."

**Design.** `DeploymentScope` is a first-class object (task, population,
setting, modality, subgroup key). The gate scores the system *on that scope*,
and the license is only valid for that scope. A benchmark score from a
different task is structurally unusable.

## 2. Why stratified scope verification (the subgroup floor)

**Evidence.** 52 of 200 papers address safety/bias/fairness; the SAGE paper
(2026) and the postpartum-depression literature make explicit that minority
subgroups are systematically under-served by risk models. An overall accuracy
of 0.85 can coexist with 0.74 accuracy on a 20% sub-population — exactly the
pattern our simulated system exhibits.

**Design.** `scope_verification` computes per-subgroup accuracy with a
minimum-subgroup-size guard (so a 3-sample "subgroup" can't veto), and flags
any subgroup below floor. The gate can then either reject or, more usefully,
issue a **restricted license** that confines the weak subgroup to
human-in-the-loop — a decision a single scalar benchmark cannot express.

## 3. Why calibration is mandatory

**Evidence.** The safety-gated mental-health backend (2026) and the
"Relevance is not enough" retrieval paper both argue that for consequential
health questions, *what the system says it is sure of* matters as much as what
it says. The Open-Box Fallacy paper's 53-point gap between understanding and
correction is, operationally, a calibration/uncertainty problem: a clinician
who cannot trust the confidence cannot override the failure.

**Design.** `calibration` computes ECE and fails the gate above a threshold.
A system that is accurate but confidently wrong on the tail is rejected —
accuracy alone is not a deployment criterion.

## 4. Why a safety audit with a refusal channel

**Evidence.** The safety-gated backend (2026) blocks unsafe *generations*; the
responsible-evaluation paper (arXiv:2602.00065) and the mental-health
systematic review (2026) both treat self-harm ideation as the canonical
high-risk case where an automated answer is never acceptable.

**Design.** `Prediction` carries an `action` channel (`answer` / `refuse` /
`escalate`). `safety_audit` runs a high-risk battery and requires 100%
safe handling — an `answer` on a self-harm input is an automatic REJECT. This
is the component that fails the `unsafe` simulated system in the demo.

## 5. Why the monitoring plan is a *component*, and why revocation is automatic

**Evidence.** The Open-Box Fallacy paper: only **9.0%** of FDA-approved
AI/ML device documents contained a prospective post-market surveillance study.
The Misattribution Gap paper (arXiv:2605.22842) shows deployed agentic
systems failing in ways that get misdiagnosed *after* release because nothing
was watching.

**Design.** Two parts. First, `monitoring_plan` is a mandatory gate
component: a system with no executable post-market plan cannot be licensed —
the plan must name its thresholds, horizon, baseline, and revocation trigger.
Second, `DriftMonitor` *executes* that plan: each live batch is compared to
the validation baseline via Population Stability Index (distribution drift)
and a live error-rate excess (performance drift). Crossing either threshold
**revokes the license automatically**, with the reason recorded on the audit
chain. The demo's Stage 3 shows a license going ACTIVE → REVOKED with no
human in the loop.

## 6. Why contestability is a hash chain

**Evidence.** The Open-Box Fallacy standard requires authorization to be
"accountable, contestable, and revocable" — governed "through credentials,
monitoring, liability, appeal, and revocation rather than mechanism-level
explanation."

**Design.** `AuditTrail` is append-only and hash-chained (each entry's SHA-256
covers the previous hash + canonical content). Every event — evaluation,
decision, license issuance, each live batch, revocation — is on the chain.
`AuditTrail.verify()` recomputes the whole history offline; the test suite
proves that tampering with any single entry breaks verification. A reviewer
(or a regulator, or a patient's advocate) can contest any decision by
re-verifying the chain and inspecting the exact inputs that produced it.

## 7. Why an independent check

**Evidence.** The Open-Box Fallacy standard requires authorization to be
"independently checkable." Self-attestation — the evaluator grading its own
work — is the classic failure of internal QA.

**Design.** `independent_check` is a *separately written* re-implementation
that recomputes accuracy, the Wilson CI, and ECE from raw (sample, prediction)
pairs and compares them to the primary evaluator's reported numbers.
Disagreement fails the component. It is deliberately non-mandatory (a
mismatch is a strong signal to investigate) while the four technical
components are hard gates — mirroring how real governance treats
"reproducibility" vs "safety."

## 8. Why the VCS is a weighted, transparent aggregate

**Evidence.** The Open-Box Fallacy paper proposes Verification Coverage as "a
metric that should sit beside capability scores in model cards, leaderboards,
and regulatory disclosures" — i.e., a *reportable number* with a
minimum-composition rule.

**Design.** VCS = 100 × Σ wᵢ·scoreᵢ with fixed, documented weights
(scope 0.30, safety 0.20, calibration 0.15, independent check 0.15,
monitoring 0.10, contestability 0.10). The score is a *summary*; the
decision is driven by the minimum-composition rule (any mandatory failure →
REJECT), so a high VCS can never paper over a failed safety audit. Every
component's details are serialized into the decision and the audit trail, so
the number is always explainable to the person contesting it.

## 9. Limitations (stated honestly)

- The demo system is a **simulated** postpartum-depression predictor
  (deterministic, stdlib-only) — a test double, not a clinical model. The
  gate is model-agnostic; wiring a real model is a wrapper around
  `predict(features) -> Prediction`.
- Drift detection uses PSI + error excess on subgroup composition; richer
  monitors (per-feature drift, outcome-based drift, covariate shift tests)
  slot into the same `DriftMonitor` interface.
- The safety battery is a fixed list of high-risk cases; a real deployment
  would maintain a growing, versioned battery (the `SafetyCase` type is
  designed for that).
- This is a *reference implementation of a governance standard*, not a
  regulatory submission. Thresholds are defaults, meant to be set by the
  governing body per scope.