import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


SESSION_TTL_MINUTES = _positive("SESSION_TTL_MINUTES", _int_env("SESSION_TTL_MINUTES", 60))
MAX_FAILED_LOGIN_ATTEMPTS = _positive(
    "MAX_FAILED_LOGIN_ATTEMPTS", _int_env("MAX_FAILED_LOGIN_ATTEMPTS", 5)
)
LOCKOUT_MINUTES = _positive("LOCKOUT_MINUTES", _int_env("LOCKOUT_MINUTES", 15))
USERS_DB_PATH = os.getenv("USERS_DB_PATH", "storage/users.db")
APP_LOG_DIR = os.getenv("APP_LOG_DIR", "storage/logs")
STREAMLIT_APP_URL = os.getenv(
    "STREAMLIT_APP_URL", "https://stock-anomaly-detector-tomas.streamlit.app/"
)

if ENVIRONMENT == "production":
    USERS_DB_PATH = _required_env("USERS_DB_PATH")
    APP_LOG_DIR = _required_env("APP_LOG_DIR")
    STREAMLIT_APP_URL = _required_env("STREAMLIT_APP_URL")
