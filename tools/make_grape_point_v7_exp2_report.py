from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_s_grape_point_v7_exp2.yml"


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    if "--report-mode" not in sys.argv:
        sys.argv.extend(["--report-mode", "point_v7_exp2_full"])
    if "--reference-label" not in sys.argv:
        sys.argv.extend(["--reference-label", "baseline_replay"])
    if "--report-title" not in sys.argv:
        sys.argv.extend(["--report-title", "point_v7_exp2 中文结论"])
    from make_grape_point_v2_report import main as report_main

    report_main()


if __name__ == "__main__":
    main()
