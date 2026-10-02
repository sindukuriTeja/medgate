# Innovation Problem Statement

## MedGate: A Calibrated Verification Gate for Healthcare AI Deployment

**Derived from a systematic scan of 200 recent arXiv papers matching "AI with health care"
(search: `query=ai+with+health+care`, ordered by announcement date, size 200).**

---

## 1. What the 200-paper corpus shows

Theme analysis of the corpus (keyword+abstract scan, see `docs/literature_map.csv`):

| Theme | Papers |
|---|---|
| Patient communication / patient-facing AI | 97 |
| Diagnosis / risk prediction | 65 |
| Mental health / psychiatry | 53 |
| Safety, bias, fairness, hallucination | 52 |
| Clinical evaluation / feasibility / RCT | 52 |
| LLMs / foundation models | 47 |
| Explainability / trust | 38 |
| Privacy / federated / de-identification | 28 |
| Cost / health economics / utilization | 28 |
| Medical imaging | 23 |
| EHR / clinical data | 16 |

**The field's center of gravity is capability**: better models (LLMs, imaging, EHR),
better predictions, better patient communication, better evaluations of those models.

## 2. The gap

Within the same corpus, a distinct and growing cluster of papers attacks the *deployment*
problem — not model capability, but **when a healthcare AI system is actually allowed to
be used on real patients**:

- **"The Open-Box Fallacy: Why AI Deployment Needs a Calibrated Verification Regime"**
  (arXiv:2605.10601, 2026) — argues authorization should be *domain-scoped, independently
  checkable, monitored after release, accountable, contestable, and revocable*. It proposes
  a **Verification Coverage** standard (six components, minimum-composition rule) to sit
  beside capability scores in model cards and regulatory disclosures. Two hard data points:
  a **53-percentage-point gap** between understanding internal representations and
  correcting outputs, and a scoping review finding only **9.0% of FDA-approved AI/ML
  device documents** contained a prospective post-market surveillance study.
- **"Responsible Evaluation of AI for Mental Health"** (arXiv:2602.00065, 2026) — evaluation
  frameworks for high-stakes mental-health AI.
- **"A Safety-Gated Multimodal AI Backend for Mental-Health Support"** (2026) — safety
  gating exists *inside* a generator (blocking unsafe generations), not as a deployment
  authorization layer.
- **"The Misattribution Gap: When Memory Poisoning Looks Like Model Failure in Agentic AI
  Systems"** (arXiv:2605.22842, 2026) — deployed agentic systems fail in ways that are
  misdiagnosed after release, because there is no structured post-market verification.
- **"Deployment and Evaluation of an EHR-integrated LLM Tool to Triage Surgical Patients"**
  (arXiv:2603.17234, 2026) and **"Evaluating AI Generated Summaries for Cancer Patients"**
  (2026) — each deployment builds its own bespoke evaluation; nothing reusable.
- **"A Causal Inference Approach for Evaluating Diagnostic Tests and AI-Enabled Medical
  Devices"** (2026) — evaluation methodology exists, but as an offline analysis, not a
  gate.
- **"Relevance is not enough"** (2026) — for consequential health questions, retrieval
  relevance alone is insufficient; answers need communication-level verification.

**The gap, precisely stated:**

> The literature has converged on *what* a deployment gate should be (the Open-Box
> Fallacy's Verification Coverage: scope-bound, calibrated, safety-audited,
> post-market-monitored, contestable, independently checkable) — but **no working,
> executable implementation exists**. Every deployment in the corpus rolls its own
> ad-hoc evaluation; the proposed standard is a paper, not software. There is no
> reference implementation that (a) computes a Verification Coverage score for a
> concrete system + concrete deployment scope, (b) issues a revocable, domain-scoped
> authorization decision, and (c) monitors the deployed system and revokes
> authorization when post-market drift exceeds a threshold.

## 3. The innovation problem (one paragraph)

**Build an executable "calibrated verification gate" for healthcare AI deployment.**
Given (i) a healthcare AI system and (ii) a proposed deployment scope (task, population,
setting), the gate must produce:

1. **Scope-bound verification** — performance measured on the *exact* proposed scope,
   with statistical confidence intervals (not generic benchmark scores).
2. **Calibration as a first-class requirement** — the system's confidence must be
   calibrated within the scope (uncertainty the clinician can trust).
3. **Safety-critical failure audit** — a battery of high-risk inputs; the system must
   refuse or escalate, never answer unsafely, on them.
4. **Post-market monitoring plan with automatic revocation** — a drift detector over
   live inputs/outcomes; authorization is a *license*, revocable when drift or
   error-rate thresholds are crossed (directly targeting the 9% post-market
   surveillance gap).
5. **Contestability / audit trail** — every gated decision is logged with cryptographic
   provenance (hash-chained), so any authorization can be audited and contested.
6. **Independent checkability** — a second, rule-based verifier cross-checks the
   primary evaluator (no self-attestation).

The gate outputs a **Verification Coverage Score (VCS, 0–100)** and a decision:
`APPROVE`, `APPROVE_WITH_CONDITIONS` (scope restrictions), or `REJECT` — plus a
revocable license.

## 4. Success criteria for the solution

- A working Python reference implementation (stdlib + pytest only, deterministic,
  no network, no external ML dependencies).
- A demo that runs the full lifecycle on a simulated healthcare system with a known
  blind spot: gate issues a **restricted** authorization, post-market drift is
  detected, and the license is **revoked** — demonstrating all six components.
- Automated tests covering each verification component and the end-to-end gate.
- A report mapping every design decision back to the corpus evidence above.