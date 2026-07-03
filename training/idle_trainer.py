"""Local idle-training policy for Oki.

This module does not silently train by default. It evaluates whether the personal
machine is in a safe training window, then either prints the command that would
run or starts it when --execute is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_POLICY_PATH = Path("training/local_idle_policy.json")


@dataclass(frozen=True)
class IdleDecision:
    allowed: bool
    reasons: list[str]
    command: list[str]


def load_policy(path: Path | str = DEFAULT_POLICY_PATH) -> dict:
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing idle training policy: {policy_path}")
    return json.loads(policy_path.read_text(encoding="utf-8"))


def evaluate_idle_training(policy: dict, config_path: str) -> IdleDecision:
    reasons: list[str] = []
    now = datetime.now()
    quiet_hours = policy.get("quiet_hours", {})
    start_hour = int(quiet_hours.get("start", 0))
    end_hour = int(quiet_hours.get("end", 24))
    if not _hour_in_window(now.hour, start_hour, end_hour):
        reasons.append(f"outside quiet training window {start_hour}:00-{end_hour}:00")

    if policy.get("require_ac_power", True) and _on_battery_power():
        reasons.append("machine appears to be on battery power")

    if policy.get("require_user_opt_in", True) and os.getenv("OKI_ALLOW_LOCAL_TRAINING") != "1":
        reasons.append("OKI_ALLOW_LOCAL_TRAINING=1 is required")

    command = ["python", "training/train_qlora.py", "--config", config_path]
    return IdleDecision(allowed=not reasons, reasons=reasons, command=command)


def run_idle_training(policy_path: Path, config_path: str, execute: bool) -> int:
    policy = load_policy(policy_path)
    decision = evaluate_idle_training(policy, config_path)
    print("Local idle training policy:")
    print(f"- allowed: {decision.allowed}")
    if decision.reasons:
        for reason in decision.reasons:
            print(f"- blocked: {reason}")
    print("- command: " + " ".join(decision.command))
    if not execute:
        print("Dry run only. Pass --execute after opting in to start training.")
        return 0
    if not decision.allowed:
        return 1
    completed = subprocess.run(decision.command, check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Oki fine-tuning only during local idle windows")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--config", default="training/qlora_config.json")
    parser.add_argument("--execute", action="store_true", help="Actually launch training when policy allows it")
    args = parser.parse_args(argv)
    return run_idle_training(Path(args.policy), args.config, args.execute)


def _hour_in_window(hour: int, start: int, end: int) -> bool:
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _on_battery_power() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return False
    # Windows BatteryStatus 1 means discharging. Empty output usually means desktop/no battery.
    return output == "1"


if __name__ == "__main__":
    raise SystemExit(main())

