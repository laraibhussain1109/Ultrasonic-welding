"""Serial control for an ESP32 reject/fail output pin."""

from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass


DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_S = 0.2
DEFAULT_HANDSHAKE_TIMEOUT_S = 2.0
DEFAULT_CONNECT_SETTLE_S = 1.5
PYSERIAL_INSTALL_HINT = "Install pyserial with: python -m pip install pyserial (or reinstall the app with: python -m pip install -e .[industrial])"
ESP32_PORT_KEYWORDS = ("cp210", "ch340", "ch910", "silicon labs", "usb serial", "esp32", "wch")


@dataclass(frozen=True)
class ESP32OutputConfig:
    """Configuration for the ESP32 serial fail-output controller."""

    port: str | None = None
    baudrate: int = DEFAULT_BAUDRATE
    enabled: bool = True
    connect_settle_s: float = DEFAULT_CONNECT_SETTLE_S
    scan_all_ports: bool = True
    require_handshake: bool = True

    @classmethod
    def from_env(cls) -> "ESP32OutputConfig":
        enabled_value = os.getenv("BLOWER_ESP32_ENABLED", "1").strip().lower()
        enabled = enabled_value not in {"0", "false", "no", "off", "disabled"}
        scan_value = os.getenv("BLOWER_ESP32_SCAN_ALL", "1").strip().lower()
        scan_all_ports = scan_value not in {"0", "false", "no", "off", "disabled"}
        handshake_value = os.getenv("BLOWER_ESP32_REQUIRE_HANDSHAKE", "1").strip().lower()
        require_handshake = handshake_value not in {"0", "false", "no", "off", "disabled"}
        return cls(
            port=os.getenv("BLOWER_ESP32_PORT") or autodetect_port(),
            baudrate=int(os.getenv("BLOWER_ESP32_BAUD", str(DEFAULT_BAUDRATE))),
            enabled=enabled,
            connect_settle_s=float(os.getenv("BLOWER_ESP32_SETTLE", str(DEFAULT_CONNECT_SETTLE_S))),
            scan_all_ports=scan_all_ports,
            require_handshake=require_handshake,
        )


def _serial_list_ports() -> list[str]:
    """Return serial ports reported by pyserial, prioritising USB/ESP32 devices."""

    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return []
    except Exception:
        return []

    ports = list(list_ports.comports())

    def score(port) -> int:
        text = " ".join(
            str(value).lower()
            for value in (port.device, port.description, port.manufacturer, port.hwid)
            if value
        )
        return 0 if any(keyword in text for keyword in ESP32_PORT_KEYWORDS) else 1

    return [port.device for port in sorted(ports, key=score)]


def candidate_ports(preferred_port: str | None = None) -> list[str]:
    """Return candidate serial ports with the preferred port first."""

    candidates: list[str] = []
    if preferred_port:
        candidates.append(preferred_port)
    candidates.extend(_serial_list_ports())
    if sys.platform.startswith("win"):
        candidates.extend(f"COM{index}" for index in range(1, 21))
    elif sys.platform == "darwin":
        candidates.extend(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB_USBtoUART*") + glob.glob("/dev/cu.wchusbserial*"))
    else:
        candidates.extend(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def autodetect_port() -> str | None:
    """Return a likely ESP32 serial port for the current OS, or ``None``."""

    ports = candidate_ports()
    return ports[0] if ports else None


class ESP32FailOutput:
    """Best-effort serial client for driving the ESP32 fail output.

    The matching firmware accepts newline-terminated commands:
    ``FAIL`` drives the configured GPIO HIGH, while ``PASS``, ``LOW``, and
    ``STANDBY`` drive it LOW.
    """

    def __init__(self, config: ESP32OutputConfig | None = None) -> None:
        self.config = config or ESP32OutputConfig.from_env()
        self._serial = None
        self.connected_port: str | None = None
        self.last_error: str | None = None
        self.last_response: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and bool(getattr(self._serial, "is_open", False))

    def connect(self) -> bool:
        if not self.config.enabled:
            self.last_error = "ESP32 output disabled"
            return False
        if self.is_connected:
            return True
        ports = candidate_ports(self.config.port) if self.config.scan_all_ports else ([self.config.port] if self.config.port else [])
        if not ports:
            self.last_error = "No ESP32 serial port found; set BLOWER_ESP32_PORT"
            return False

        errors: list[str] = []
        for port in ports:
            if self._open_port(port):
                return True
            if self.last_error:
                errors.append(self.last_error)
        self.last_error = "; ".join(errors[:4])
        if len(errors) > 4:
            self.last_error += f"; ... tried {len(errors)} ports"
        return False

    def _open_port(self, port: str) -> bool:
        try:
            import serial  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            self.last_error = PYSERIAL_INSTALL_HINT
            return False

        try:
            self._serial = serial.Serial(port, self.config.baudrate, timeout=DEFAULT_TIMEOUT_S)
            self.connected_port = port
            self.last_error = None
            if self.config.connect_settle_s > 0:
                time.sleep(self.config.connect_settle_s)
            if self.config.require_handshake and not self._verify_firmware():
                failed_response = self.last_response or "no response"
                self.last_error = f"{port}: opened but ESP32 firmware handshake failed ({failed_response})"
                self.close()
                return False
            return True
        except Exception as exc:
            self._serial = None
            self.connected_port = None
            self.last_error = f"{port}: {exc}"
            return False

    def _verify_firmware(self) -> bool:
        if self._serial is None:
            return False
        try:
            if hasattr(self._serial, "reset_input_buffer"):
                self._serial.reset_input_buffer()
            self._serial.write(b"PING\n")
            self._serial.flush()
            deadline = time.monotonic() + DEFAULT_HANDSHAKE_TIMEOUT_S
            responses: list[str] = []
            while time.monotonic() < deadline:
                raw = self._serial.readline()
                if not raw:
                    continue
                response = raw.decode("utf-8", errors="replace").strip()
                if not response:
                    continue
                responses.append(response)
                if response == "PONG" or response.startswith("ESP32_FAIL_OUTPUT_READY"):
                    self.last_response = response
                    return True
            self.last_response = "; ".join(responses[-3:]) if responses else None
            return False
        except Exception as exc:
            self.last_response = f"handshake error: {exc}"
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
            failed_port = self.connected_port or "unknown port"
            self.last_error = f"{failed_port}: serial write failed: {exc}"
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
            self.connected_port = None
