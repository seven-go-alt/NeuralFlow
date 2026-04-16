from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SkillDefinition:
    name: str
    description: str
    tool_name: str
    read_only: bool = True


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        tool_name: str | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        self._skills[name] = SkillDefinition(
            name=name,
            description=description,
            tool_name=tool_name or name,
            read_only=read_only,
        )

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def get_allowed_skills(self, whitelist: list[str] | None) -> list[SkillDefinition]:
        if not whitelist:
            return []
        return [self._skills[name] for name in whitelist if name in self._skills]


skill_registry = SkillRegistry()
skill_registry.register("memory", "查询长期记忆与历史摘要")
skill_registry.register("planner", "生成任务拆分与执行计划")
skill_registry.register("python", "执行 Python 相关辅助能力")
skill_registry.register("filesystem", "处理文件读写与项目结构信息")
