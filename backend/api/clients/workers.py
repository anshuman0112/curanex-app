"""HTTP client for the model workers."""
import httpx

from api.core.config import get_settings


class WorkerError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def predict(worker: str, model_id: str, inputs: dict) -> dict:
    s = get_settings()
    try:
        r = httpx.post(f"{s.worker_url(worker).rstrip('/')}/predict/{model_id}", json={"inputs": inputs},
                       timeout=s.worker_timeout_s)
    except httpx.HTTPError as e:
        raise WorkerError(f"worker-{worker} unreachable") from e
    if r.status_code == 422:
        detail = r.json().get("detail")
        errors = detail.get("errors") if isinstance(detail, dict) else [str(detail)]
        raise WorkerError("; ".join(errors), 422)
    if r.status_code != 200:
        raise WorkerError(f"worker-{worker} returned {r.status_code}", r.status_code)
    return r.json()
