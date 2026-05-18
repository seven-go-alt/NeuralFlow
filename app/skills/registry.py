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
skill_registry.register("python_exec", "执行 Python 代码并返回输出结果")
skill_registry.register("file_read", "读取沙箱目录中的文件内容")
skill_registry.register("file_write", "将内容写入沙箱目录中的文件", read_only=False)
skill_registry.register("file_list", "列出沙箱目录中的文件和子目录")
skill_registry.register("terminal", "在本地终端中执行 shell 命令（Linux/macOS），返回标准输出、标准错误和退出码", read_only=False)
