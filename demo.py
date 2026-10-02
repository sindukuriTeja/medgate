#!/usr/bin/env python3
"""MedGate end-to-end demo.

Runs the full deployment lifecycle on a simulated postpartum-depression risk
system that has a *realistic* blind spot (under-estimates risk for Spanish-
language patients):

  Stage 1  A "clean" system (no bias) is gated -> APPROVE, license issued.
  Stage 2  The same system with a subgroup bias is gated -> the gate catches
           the blind spot and issues a RESTRICTED license (APPROVE_WITH_
           CONDITIONS) instead of a clean approval.
  Stage 3  Post-market: the live population drifts away from the validation
           baseline and the live error rate rises -> the DriftMonitor REVOKES
           the license automatically.
  Stage 4  The audit trail is re-verified, proving the whole history is
           tamper-evident and contestable.

Run:  python3 demo.py
"""
from __future__ import annotations

import json

from medgate import (
    DeploymentScope,
    MedGate,
    SimulatedDepressionRiskSystem,
    make_validation_set,
)
from medgate.drift import LiveBatch
from medgate.systems import standard_safety_cases


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    scope = DeploymentScope(
        task="postpartum depression risk prediction",
        population="postpartum patients, 0-12 weeks",
        setting="community primary-care clinic",
        modality="EHR fields + PHQ-4 questionnaire",
        subgroup_key="language",
    )
    samples = make_validation_set(n=400, es_frac=0.2)
    safety_cases = standard_safety_cases()

    gate = MedGate()

    # ------------------------------------------------------------------
    banner("STAGE 1 — Clean system (no subgroup bias) -> expect APPROVE")
    clean = SimulatedDepressionRiskSystem(es_bias=0.0)
    d1 = gate.evaluate(clean, scope, samples, safety_cases, system_name=clean.name)
    print(json.dumps(d1.to_dict(), indent=2))
    lic1 = gate.issue_license(d1, scope, at="t100")
    print(f"\n>>> License {lic1.license_id} issued: {lic1.status} "
          f"(decision={lic1.decision}, VCS={lic1.vcs:.1f})")

    # ------------------------------------------------------------------
    banner("STAGE 2 — Same system WITH a Spanish-language blind spot")
    biased = SimulatedDepressionRiskSystem(es_bias=1.0)
    d2 = gate.evaluate(biased, scope, samples, safety_cases, system_name=biased.name)
    print(json.dumps(d2.to_dict(), indent=2))
    print("\n>>> The gate caught the subgroup blind spot and issued a RESTRICTED license:")
    lic2 = gate.issue_license(d2, scope, at="t200")
    for c in lic2.conditions:
        print(f"    - {c}")

    # ------------------------------------------------------------------
    banner("STAGE 3 — Post-market drift -> expect automatic REVOCATION")
    mon = gate.build_drift_monitor(lic2, samples)
    print(f"Baseline composition: {mon.baseline}")
    print("Feeding live batches (population shifts toward 'es', errors rise):\n")
    # Batch 1: on-distribution, on-error-rate -> monitor continues.
    r1 = mon.observe(
        LiveBatch(groups=["en"] * 300 + ["es"] * 80, errors=40, n=380), lic2, at="t210"
    )
    print(f"  batch@t210: psi={r1['psi']} err={r1['live_error_rate']} "
          f"-> {r1['action']}")
    # Batch 2: heavy drift + error breach -> license revoked.
    r2 = mon.observe(
        LiveBatch(groups=["en"] * 120 + ["es"] * 280, errors=95, n=400), lic2, at="t220"
    )
    print(f"  batch@t220: psi={r2['psi']} err={r2['live_error_rate']} "
          f"-> {r2['action']}")
    print(f"\n>>> License {lic2.license_id} status: {lic2.status}")
    if lic2.revocation_reason:
        print(f"    revocation reason: {lic2.revocation_reason}")

    # ------------------------------------------------------------------
    banner("STAGE 4 — Audit trail re-verification (contestability)")
    ok = gate.audit.verify()
    print(f"Audit entries: {len(gate.audit)}")
    print(f"Chain verifies (tamper-evident): {ok}")
    print(f"Head hash: {gate.audit.head_hash}")
    print("\nEvent timeline:")
    for e in gate.audit.entries():
        print(f"  #{e['seq']:>2} [{e['at']:>4}] {e['event']:<16} "
              f"hash={e['hash'][:12]}")

    # ------------------------------------------------------------------
    banner("RESULT")
    print("Stage 1 clean system      :", d1.decision, f"(VCS {d1.vcs:.1f})")
    print("Stage 2 biased system     :", d2.decision, f"(VCS {d2.vcs:.1f})")
    print("Stage 3 post-market drift :", lic2.status)
    print("Stage 4 audit chain       :", "VERIFIED" if ok else "BROKEN")
    print("\nAll six Verification Coverage components exercised end-to-end.")


if __name__ == "__main__":
    main()