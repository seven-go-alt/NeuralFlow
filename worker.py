from celery import Celery

from app.config import get_settings
from app.memory.long_term import LongTermMemory
from app.memory.summarizer import Summarizer

settings = get_settings()

celery_app = Celery(
    "neuralflow",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


@celery_app.task
def compress_and_archive(session_id: str, history_text: str) -> str:
    summarizer = Summarizer()
    summary = summarizer.summarize(history_text)

    long_term = LongTermMemory()
    long_term.save_summary(summary, {"session_id": session_id})

    return f"Archived summary for session={session_id}"
