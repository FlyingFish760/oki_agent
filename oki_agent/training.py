from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json

from oki_agent.memory import MemoryStore
from oki_agent.persona import PersonaCard


@dataclass(frozen=True)
class TrainingJob:
    manifest_path: Path
    dataset_path: Path


class LocalTrainingQueue:
    def __init__(self, root: Path) -> None:
        self.root = root / ".oki" / "training"
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare_persona_job(self, persona: PersonaCard, memory: MemoryStore) -> TrainingJob:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dataset_path = self.root / f"persona_samples_{stamp}.jsonl"
        manifest_path = self.root / f"train_lora_{stamp}.json"
        samples = self._build_samples(persona, memory)
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "base_model": "qwen3.6-35b-a3b",
            "training_mode": "local_idle_lora",
            "dataset": str(dataset_path),
            "adapter_output": str(self.root / f"adapters{stamp}"),
            "recommended_config": {
                "method": "QLoRA",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "rank": 8,
                "learning_rate": 0.0001,
                "epochs": 1,
                "activation": "manual_after_eval",
            },
            "safety": {
                "local_only": True,
                "requires_user_approval_before_activation": True,
                "rollback_previous_adapter": True,
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return TrainingJob(manifest_path=manifest_path, dataset_path=dataset_path)

    def _build_samples(self, persona: PersonaCard, memory: MemoryStore) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = [
            {
                "messages": [
                    {"role": "system", "content": persona.system_prompt()},
                    {"role": "user", "content": "我今天有点乱，不知道先做什么。"},
                    {
                        "role": "assistant",
                        "content": "我们先把事情放到桌面上。告诉我最卡住的一件事，我会帮你拆成下一步能做的小动作。",
                    },
                ]
            }
        ]
        for item in memory.list_recent(limit=20):
            samples.append(
                {
                    "messages": [
                        {"role": "system", "content": persona.system_prompt()},
                        {"role": "user", "content": f"记住这点：{item.content}"},
                        {"role": "assistant", "content": "收到，我会把它作为你的长期偏好来参考，但需要时也会向你确认。"},
                    ],
                    "metadata": {"memory_kind": item.kind, "importance": item.importance},
                }
            )
        return samples
