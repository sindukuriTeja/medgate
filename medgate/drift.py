"""Post-market drift monitoring with automatic license revocation.

The license is a *permission*, not a certificate. After deployment, live
batches of (input, system-output, outcome) are fed to the DriftMonitor. It
compares the live population against the validation baseline and revokes the
license when drift or live error exceeds the thresholds fixed in the
monitoring plan. This is the component that directly attacks the finding that
only ~9% of FDA-approved AI/ML device documents contained a prospective
post-market surveillance study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import composition, population_stability_index
from .models import License


@dataclass
class LiveBatch:
    """One post-market observation batch.

    ``groups``    — the subgroup labels of live inputs (for distribution drift)
    ``errors``    — number of confirmed errors in this batch
    ``n``         — total live samples in the batch
    """

    groups: List[str]
    errors: int
    n: int


@dataclass
class DriftConfig:
    psi_threshold: float = 0.25
    error_excess_threshold: float = 0.15
    baseline_error_rate: float = 0.10  # error rate observed during verification


class DriftMonitor:
    """Executes the monitoring plan attached to a license.

    ``on_event`` (optional) is called with each batch report so the
    governance layer can hash-chain post-market observations into the audit
    trail — revocations included.
    """

    def __init__(
        self,
        cfg: DriftConfig,
        baseline_composition: Dict[str, float],
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.baseline = dict(baseline_composition)
        self.on_event = on_event
        self.history: List[Dict] = []

    def observe(self, batch: LiveBatch, license_obj: License, at: str = "t?") -> Dict:
        """Process one live batch; may revoke the license. Returns the report."""
        live_comp = composition(batch.groups) or {}
        keys = sorted(set(self.baseline) | set(live_comp))
        psi = population_stability_index(
            [self.baseline.get(k, 0.0) for k in keys],
            [live_comp.get(k, 0.0) for k in keys],
        )
        live_err = batch.errors / batch.n if batch.n else 0.0
        err_excess = live_err - self.cfg.baseline_error_rate

        report = {
            "at": at,
            "n": batch.n,
            "psi": round(psi, 4),
            "live_error_rate": round(live_err, 4),
            "error_excess": round(err_excess, 4),
            "live_composition": {k: round(live_comp.get(k, 0.0), 4) for k in keys},
            "drifted": psi > self.cfg.psi_threshold,
            "error_breach": err_excess > self.cfg.error_excess_threshold,
            "action": "monitor",
        }

        if license_obj.status == "ACTIVE" and (report["drifted"] or report["error_breach"]):
            reasons = []
            if report["drifted"]:
                reasons.append(
                    f"population drift PSI={report['psi']} > {self.cfg.psi_threshold}"
                )
            if report["error_breach"]:
                reasons.append(
                    f"live error excess {report['error_excess']} > {self.cfg.error_excess_threshold}"
                )
            license_obj.revoke("; ".join(reasons), at=at)
            report["action"] = "REVOKED"

        self.history.append(report)
        if self.on_event is not None:
            self.on_event({
                "license_id": license_obj.license_id,
                "license_status": license_obj.status,
                **report,
            })
        return report

    def summary(self) -> Dict:
        return {
            "batches": len(self.history),
            "max_psi": max((h["psi"] for h in self.history), default=0.0),
            "revocations": sum(1 for h in self.history if h["action"] == "REVOKED"),
        }