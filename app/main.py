from fastapi import FastAPI, Request
from dotenv import load_dotenv
load_dotenv()

from app.routes.agent_routes import router
from fastapi.middleware.cors import CORSMiddleware
import logging
import logging.config
import time

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s - %(message)s",
        },
        "access": {
            "format": "%(asctime)s %(levelname)s %(name)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        }
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "app": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")

app = FastAPI()

# Allow all CORS for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# This is the correct placement for non-prefixed routes
# They are added directly to the main app instance
@app.get("/")
async def read_root():
    logger.info("Root endpoint hit")
    return {"message": "Welcome to my API! 🎉"}

# Add an additional, optional endpoint to check the service status
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Simple request logging middleware (method, path, status, duration)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "HTTP %s %s -> %s (%d ms)",
        request.method,
        request.url.path,
        getattr(response, "status_code", "-"),
        duration_ms,
    )
    return response


# Register router AFTER the app-level routes
# The prefix is applied here, so all routes in 'router'
# will be available at /api/...
app.include_router(router, prefix="/api")