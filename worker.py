import asyncio

from celery import Celery

from app.config import get_settings
from app.core.llm import LLMClient
from app.memory.long_term import LongTermMemory
from app.memory.summarizer import Summarizer

settings = get_settings()

celery_app = Celery(
    "neuralflow",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


@celery_app.task
def compress_and_archive(
    session_id: str,
    history_text: str | None = None,
    messages: list[dict[str, str]] | None = None,
    long_term_memory: LongTermMemory | None = None,
) -> str:
    llm_client = LLMClient()
    summarizer = Summarizer(llm_client=llm_client)

    if messages is not None:
        loop = asyncio.new_event_loop()
        try:
            summary = loop.run_until_complete(
                summarizer.summarize_messages_async(session_id=session_id, messages=messages)
            )
        finally:
            loop.close()
        message_count = len(messages)
    else:
        summary = summarizer.summarize(history_text or "")
        message_count = 0

    long_term = long_term_memory or LongTermMemory()
    long_term.save_summary(
        summary,
        {
            "session_id": session_id,
            "message_count": message_count,
            "source": "archive",
        },
    )

    return f"Archived summary for session={session_id}"
