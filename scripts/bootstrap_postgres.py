from __future__ import annotations

from config import DATABASE_URL
from repositories.sqlalchemy_adapter import SqlAlchemyAdapter
from repositories.sqlalchemy_migrations import ensure_domain_schema

try:
    from sqlalchemy import text
except Exception:  # pragma: no cover - optional import during docs-only flows
    text = None  # type: ignore[assignment]


def main() -> None:
    adapter = SqlAlchemyAdapter(database_url=DATABASE_URL)
    status = adapter.ping()
    if not status.ok:
        raise RuntimeError(f"Database ping failed: {status.message}")

    engine = adapter.get_engine()
    if engine is None:
        raise RuntimeError("SQLAlchemy engine unavailable")

    version = ensure_domain_schema(engine)

    if text is not None:
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(LOWER(email))",
            "CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_alert_rules_username_ticker ON alert_rules(username, ticker)",
            "CREATE INDEX IF NOT EXISTS idx_watchlists_username ON watchlists(username)",
        ]
        with engine.begin() as conn:
            for statement in index_statements:
                conn.execute(text(statement))

    print(
        f"Database bootstrap completed. dialect={status.dialect} version={version} url={DATABASE_URL}"
    )


if __name__ == "__main__":
    main()
