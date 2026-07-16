import hashlib
import os
import re
import secrets
import sqlite3
from datetime import datetime
from typing import Optional, Tuple

import bcrypt


class AuthService:
    def __init__(self, db_path: str = "users.db") -> None:
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def create_tables(self) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT,
                login_time TEXT
            )""")
        conn.commit()
        conn.close()

    def migrate_users_table(self) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        if "email" not in columns and columns:
            cursor.execute("ALTER TABLE users RENAME TO users_old")
            cursor.execute("""CREATE TABLE users (
                    username TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )""")
            cursor.execute(
                "INSERT INTO users (username, email, first_name, last_name, password, created_at) "
                'SELECT username, "", "", "", password, created_at FROM users_old'
            )
            cursor.execute("DROP TABLE users_old")
            conn.commit()
        conn.close()

    def initialize(self) -> None:
        if os.path.exists(self.db_path):
            self.migrate_users_table()
        self.create_tables()

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _legacy_hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    @classmethod
    def verify_password(cls, password: str, stored_hash: str) -> bool:
        if stored_hash.startswith("$2"):
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        return cls._legacy_hash_password(password) == stored_hash

    @staticmethod
    def is_strong_password(password: str):
        pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^\w\s]).{8,}$"
        return re.match(pattern, password)

    def is_username_available(self, username: str) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE username=?", (username,))
        result = cursor.fetchone()
        conn.close()
        return result is None

    def is_email_available(self, email: str) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE lower(email)=lower(?)", (email.strip(),))
        result = cursor.fetchone()
        conn.close()
        return result is None

    def register_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        password: str,
    ) -> Tuple[bool, Optional[str]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, first_name, last_name, password, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    username.strip(),
                    email.strip(),
                    first_name.strip(),
                    last_name.strip(),
                    self.hash_password(password),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return True, None
        except sqlite3.IntegrityError as exc:
            msg = str(exc).lower()
            if "users.username" in msg:
                return False, "Username is not available."
            if "users.email" in msg:
                return False, "Email is already registered. Try logging in."
            return False, "Registration failed due to a data conflict. Please try again."
        finally:
            conn.close()

    def authenticate_user(self, user_or_email: str, password: str) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password FROM users WHERE username=? OR email=?", (user_or_email, user_or_email)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        stored_hash = row[0]
        password_ok = self.verify_password(password, stored_hash)

        if password_ok and not stored_hash.startswith("$2"):
            # Upgrade legacy SHA-256 hash to bcrypt after successful login.
            cursor.execute(
                "UPDATE users SET password=? WHERE username=? OR email=?",
                (self.hash_password(password), user_or_email, user_or_email),
            )
            conn.commit()

        conn.close()
        return password_ok

    def get_username_by_identifier(self, user_or_email: str) -> Optional[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username FROM users WHERE username=? OR email=?", (user_or_email, user_or_email)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def create_session(self, username: str) -> str:
        session_id = secrets.token_hex(32)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, username, login_time) VALUES (?, ?, ?)",
            (session_id, username, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return session_id
