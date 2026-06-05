"""Role-based authentication helpers for the inspection UI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ITERATIONS = 260_000


@dataclass(frozen=True)
class User:
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Return a Django-style PBKDF2-SHA256 password hash."""
    if not password:
        raise ValueError("Password cannot be empty")
    salt = salt or os.urandom(12)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-SHA256 hash."""
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_text.encode("ascii"))
        expected = base64.b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations_text))
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)


class AuthStore:
    """JSON-backed user store with role support."""

    def __init__(self, path: str | Path = "config/users.json") -> None:
        self.path = Path(path)

    def _load(self) -> dict:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def authenticate(self, username: str, password: str) -> User | None:
        username = username.strip()
        if not username or not password:
            return None
        for raw_user in self._load().get("users", []):
            if raw_user.get("username") == username and verify_password(password, raw_user.get("password_hash", "")):
                return User(username=username, role=raw_user.get("role", "user"))
        return None

    def upsert_user(self, username: str, password: str, role: str = "user") -> None:
        if role not in {"admin", "user"}:
            raise ValueError("role must be 'admin' or 'user'")
        data = self._load() if self.path.exists() else {"users": []}
        users = [u for u in data.get("users", []) if u.get("username") != username]
        users.append({"username": username, "role": role, "password_hash": hash_password(password)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump({"users": users}, handle, indent=2)
            handle.write("\n")
