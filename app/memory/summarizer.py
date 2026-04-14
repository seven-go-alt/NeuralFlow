class Summarizer:
    def summarize(self, history_text: str) -> str:
        text = history_text.strip()
        if not text:
            return "空会话，无需归档。"
        if len(text) <= 240:
            return f"摘要: {text}"
        return f"摘要: {text[:237]}..."
