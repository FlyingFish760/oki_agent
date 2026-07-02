from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PersonaCard:
    name: str
    tagline: str
    voice: list[str]
    values: list[str]
    boundaries: list[str]
    response_rules: list[str]
    catchphrases: list[str]
    scenarios: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "PersonaCard":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        persona = data["persona"]
        return cls(
            name=persona["name"],
            tagline=persona["tagline"],
            voice=list(persona.get("voice", [])),
            values=list(persona.get("values", [])),
            boundaries=list(persona.get("boundaries", [])),
            response_rules=list(persona.get("response_rules", [])),
            catchphrases=list(persona.get("catchphrases", [])),
            scenarios=dict(persona.get("scenarios", {})),
        )

    def system_prompt(self) -> str:
        sections = [
            f"You are {self.name}, {self.tagline}.",
            _bullet_section("Voice", self.voice),
            _bullet_section("Values", self.values),
            _bullet_section("Boundaries", self.boundaries),
            _bullet_section("Response rules", self.response_rules),
            _bullet_section("Catchphrases to use sparingly", self.catchphrases),
            _mapping_section("Scenario behavior", self.scenarios),
        ]
        return "\n\n".join(section for section in sections if section)


def _bullet_section(title: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{title}:\n" + "\n".join(f"- {value}" for value in values)


def _mapping_section(title: str, values: dict[str, str]) -> str:
    if not values:
        return ""
    return f"{title}:\n" + "\n".join(f"- {key}: {value}" for key, value in values.items())
