#!/usr/bin/env python3
"""Load the locked checkpoint, attach E4 LoRA, and persist its exact identity."""

from __future__ import annotations

import json
from pathlib import Path

from pv_tsfm.models import Chronos2Forecaster, apply_shared_pv_lora
from pv_tsfm.models.identity import parameter_hash


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = Chronos2Forecaster.from_pretrained(
        device_map="cpu", torch_dtype="float32", local_files_only=True
    ).pipeline.model
    original_hash = parameter_hash(base, include_lora=False)
    adapted, report = apply_shared_pv_lora(base)
    if parameter_hash(adapted, include_lora=False) != original_hash:
        raise RuntimeError("base parameter hash changed while attaching LoRA")
    report["original_parameter_hash"] = original_hash
    report["peft_version"] = __import__("peft").__version__
    output = ROOT / "reports" / "lora_identity.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
