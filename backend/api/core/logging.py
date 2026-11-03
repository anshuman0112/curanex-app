"""Structured logging. Never log request bodies, messages, form values or images: they contain health data."""
import logging
import time

import structlog
from fastapi import Request


def setup_logging(level: str = "INFO", json: bool = True) -> None:
    logging.basicConfig(format="%(message)s", level=level)
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
    )


log = structlog.get_logger("curanex")


async def access_log(request: Request, call_next):
    """Method, path, status and latency only."""
    start = time.perf_counter()
    response = await call_next(request)
    log.info("request", method=request.method, path=request.url.path, status=response.status_code,
             ms=round((time.perf_counter() - start) * 1000, 1))
    return response
