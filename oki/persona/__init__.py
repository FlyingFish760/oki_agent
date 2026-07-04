"""人格系统：三层实现的运行时侧。

  1. system prompt（最轻、可动态调）—— 本模块负责
  2. 微调 adapter（最稳）—— 见 finetune/
  3. 记忆中的自我画像（缓慢演化）—— 见 memory/consolidation.py

本模块把 persona card + 用户可调旋钮（正式/活泼、话多/简洁）
组装成一段 system prompt，并预留人格漂移的检测口子。
"""

from .persona import Persona

__all__ = ["Persona"]
