from dataclasses import dataclass


@dataclass(slots=True)
class RoutedIntent:
    name: str
    requires_long_term_memory: bool


class IntentRouter:
    HISTORY_KEYWORDS = ("之前", "历史", "偏好", "记得", "上次")
    CODING_KEYWORDS = ("代码", "bug", "接口", "函数", "部署")

    def route(self, user_query: str) -> RoutedIntent:
        if any(keyword in user_query for keyword in self.HISTORY_KEYWORDS):
            return RoutedIntent(name="personal_preference", requires_long_term_memory=True)
        if any(keyword in user_query for keyword in self.CODING_KEYWORDS):
            return RoutedIntent(name="coding", requires_long_term_memory=False)
        return RoutedIntent(name="general", requires_long_term_memory=False)
