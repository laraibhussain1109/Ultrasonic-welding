"""Serial control for an ESP32 reject/fail output pin."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_S = 0.2


@dataclass(frozen=True)
class ESP32OutputConfig:
    """Configuration for the ESP32 serial fail-output controller."""

    port: str | None = None
    baudrate: int = DEFAULT_BAUDRATE
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "ESP32OutputConfig":
        enabled_value = os.getenv("BLOWER_ESP32_ENABLED", "1").strip().lower()
        enabled = enabled_value not in {"0", "false", "no", "off", "disabled"}
        return cls(
            port=os.getenv("BLOWER_ESP32_PORT") or autodetect_port(),
            baudrate=int(os.getenv("BLOWER_ESP32_BAUD", str(DEFAULT_BAUDRATE))),
            enabled=enabled,
        )


def autodetect_port() -> str | None:
    """Return a likely ESP32 serial port for the current OS, or ``None``."""

    if sys.platform.startswith("win"):
        return os.getenv("BLOWER_ESP32_PORT") or "COM3"
    candidates = [
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        "/dev/ttyACM0",
        "/dev/ttyACM1",
        "/dev/cu.usbserial-0001",
        "/dev/cu.SLAB_USBtoUART",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class ESP32FailOutput:
    """Best-effort serial client for driving the ESP32 fail output.

    The matching firmware accepts newline-terminated commands:
    ``FAIL`` drives the configured GPIO HIGH, while ``PASS``, ``LOW``, and
    ``STANDBY`` drive it LOW.
    """

    def __init__(self, config: ESP32OutputConfig | None = None) -> None:
        self.config = config or ESP32OutputConfig.from_env()
        self._serial = None
        self.last_error: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and bool(getattr(self._serial, "is_open", False))

    def connect(self) -> bool:
        if not self.config.enabled:
            self.last_error = "ESP32 output disabled"
            return False
        if self.is_connected:
            return True
        if not self.config.port:
            self.last_error = "No ESP32 serial port found; set BLOWER_ESP32_PORT"
            return False
        try:
            import serial  # type: ignore[import-not-found]

            self._serial = serial.Serial(self.config.port, self.config.baudrate, timeout=DEFAULT_TIMEOUT_S)
            self.last_error = None
            return True
        except Exception as exc:
            self._serial = None
            self.last_error = f"ESP32 serial unavailable on {self.config.port}: {exc}"
            return False

    def send(self, command: str) -> bool:
        if not self.connect():
            return False
        try:
            self._serial.write(f"{command.strip().upper()}\n".encode("ascii"))
            self._serial.flush()
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"ESP32 serial write failed: {exc}"
            self.close()
            return False

    def set_fail(self, failed: bool) -> bool:
        return self.send("FAIL" if failed else "PASS")

    def close(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        finally:
            self._serial = None
