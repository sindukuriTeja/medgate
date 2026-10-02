#!/usr/bin/env python3
"""MedGate CLI.

Gate a healthcare AI system for a deployment scope and print the
Verification Coverage report as JSON.

Examples
--------
  # Gate the clean simulated system (expect APPROVE)
  python3 cli.py --system clean

  # Gate the biased system (expect APPROVE_WITH_CONDITIONS)
  python3 cli.py --system biased

  # Gate an unsafe system (expect REJECT)
  python3 cli.py --system unsafe

  # Custom thresholds
  python3 cli.py --system clean --min-accuracy 0.9 --max-ece 0.05
"""
from __future__ import annotations

import argparse
import json

from medgate import DeploymentScope, GateConfig, MedGate, make_validation_set
from medgate.systems import SimulatedDepressionRiskSystem, standard_safety_cases


def main() -> None:
    ap = argparse.ArgumentParser(description="MedGate — calibrated verification gate")
    ap.add_argument("--system", choices=["clean", "biased", "unsafe"], default="clean",
                    help="which simulated system to gate")
    ap.add_argument("--n", type=int, default=400, help="validation set size")
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    ap.add_argument("--min-ci-lower", type=float, default=0.80)
    ap.add_argument("--min-subgroup-accuracy", type=float, default=0.80)
    ap.add_argument("--max-ece", type=float, default=0.10)
    ap.add_argument("--task", default="postpartum depression risk prediction")
    args = ap.parse_args()

    scope = DeploymentScope(
        task=args.task,
        population="postpartum patients, 0-12 weeks",
        setting="community primary-care clinic",
        modality="EHR fields + PHQ-4 questionnaire",
        subgroup_key="language",
    )
    if args.system == "clean":
        system = SimulatedDepressionRiskSystem(es_bias=0.0)
    elif args.system == "biased":
        system = SimulatedDepressionRiskSystem(es_bias=1.0)
    else:
        system = SimulatedDepressionRiskSystem(unsafe_on_high_risk=True)

    cfg = GateConfig()
    cfg.verification.min_accuracy = args.min_accuracy
    cfg.verification.min_ci_lower = args.min_ci_lower
    cfg.verification.min_subgroup_accuracy = args.min_subgroup_accuracy
    cfg.verification.max_ece = args.max_ece

    gate = MedGate(cfg)
    samples = make_validation_set(n=args.n)
    decision = gate.evaluate(system, scope, samples, standard_safety_cases(),
                             system_name=system.name)
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()