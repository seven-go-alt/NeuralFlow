from dataclasses import dataclass


@dataclass(slots=True)
class SkillDefinition:
    name: str
    description: str


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, name: str, description: str) -> None:
        self._skills[name] = SkillDefinition(name=name, description=description)

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())
