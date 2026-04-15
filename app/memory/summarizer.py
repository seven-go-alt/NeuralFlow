class Summarizer:
    def summarize(self, history_text: str) -> str:
        text = history_text.strip()
        if not text:
            return "空会话，无需归档。"
        if len(text) <= 240:
            return f"摘要: {text}"
        return f"摘要: {text[:237]}..."

    def summarize_messages(self, session_id: str, messages: list[dict[str, str]]) -> str:
        if not messages:
            return f"session={session_id}\n对话为空。"

        lines = [f"session={session_id}", "conversation:"]
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "").strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
