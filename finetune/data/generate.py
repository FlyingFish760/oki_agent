"""Compatibility entry point for the English SFT generation route.

New code should invoke ``generate_sft.py`` or ``generate_instructions.py``
directly. The former ``--mode instruction-en`` argument is accepted here so
existing commands keep working.
"""

from __future__ import annotations

import sys

try:
    from generate_sft import main as generate_sft_main
except ModuleNotFoundError:
    from .generate_sft import main as generate_sft_main


def _compatibility_args(argv: list[str]) -> list[str]:
    args = list(argv)
    if "--mode" not in args:
        return args

    mode_index = args.index("--mode")
    if mode_index + 1 >= len(args):
        raise SystemExit("--mode requires a value")
    mode = args[mode_index + 1]
    if mode != "instruction-en":
        raise SystemExit(
            "Legacy persona mode is not part of the split generators. "
            "Use generate_sft.py or generate_instructions.py."
        )
    del args[mode_index : mode_index + 2]
    return args


def main() -> None:
    generate_sft_main(_compatibility_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
