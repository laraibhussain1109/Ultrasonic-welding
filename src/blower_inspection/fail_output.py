"""ESP32 fail-output bridge.

The production ESP32 sketch exposes simple HTTP endpoints:
``/fail`` energizes the buzzer/relay output and ``/pass`` clears it.  This
module keeps those calls off the UI/inference thread so a missing WiFi link does
not stall inspection.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class FailOutputConfig:
    base_url: str = "http://192.168.4.1"
    timeout_s: float = 0.25
    enabled: bool = True


@dataclass(frozen=True)
class ManualDecision:
    """A decision entered on the ESP32 operator page."""

    sequence: int
    result: str
    sector: int | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> "ManualDecision":
        result = str(payload.get("result", "RECHECK")).upper()
        if result not in {"PASS", "FAIL", "RECHECK"}:
            raise ValueError(f"Unsupported result: {result}")
        sector_value = payload.get("sector")
        sector = int(sector_value) if sector_value is not None else None
        if result == "FAIL" and (sector is None or not 1 <= sector <= 14):
            raise ValueError("FAIL requires a sector from 1 to 14")
        if result != "FAIL":
            sector = None
        return cls(sequence=int(payload.get("sequence", 0)), result=result, sector=sector)


class ESP32FailOutputBridge:
    """Send PASS/FAIL state changes to the ESP32 buzzer/relay bridge."""

    def __init__(self, config: FailOutputConfig | None = None) -> None:
        if config is None:
            config = FailOutputConfig(
                base_url=os.environ.get("NEUROIRIS_ESP32_URL", FailOutputConfig.base_url).rstrip("/"),
                timeout_s=float(os.environ.get("NEUROIRIS_ESP32_TIMEOUT_S", FailOutputConfig.timeout_s)),
                enabled=os.environ.get("NEUROIRIS_ESP32_ENABLED", "1") not in {"0", "false", "False"},
            )
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="esp32-fail-output")
        self._last_state: bool | None = None
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def send_result(self, failed: bool) -> Future[str] | None:
        """Queue an ESP32 update when the inspection state changes."""
        if not self.config.enabled:
            return None
        if failed == self._last_state and self._last_error is None:
            return None
        self._last_state = failed
        endpoint = "fail" if failed else "pass"
        return self._executor.submit(self._call_endpoint, endpoint)

    def poll_manual_decision(self) -> Future[ManualDecision] | None:
        """Fetch the latest mobile-browser decision without blocking the UI."""
        if not self.config.enabled:
            return None
        return self._executor.submit(self._fetch_manual_decision)

    def reset(self) -> Future[str] | None:
        """Clear the output and force the next result to be sent."""
        self._last_state = None
        return self.send_result(False)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _call_endpoint(self, endpoint: str) -> str:
        url = f"{self.config.base_url}/{endpoint}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                body = response.read(200).decode("utf-8", errors="replace")
        except (OSError, urllib.error.URLError) as exc:
            self._last_error = str(exc)
            raise RuntimeError(f"ESP32 output unavailable at {url}: {exc}") from exc
        self._last_error = None
        return body

    def _fetch_manual_decision(self) -> ManualDecision:
        url = f"{self.config.base_url}/api/decision"
        request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                payload = json.loads(response.read(1024).decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self._last_error = str(exc)
            raise RuntimeError(f"ESP32 decision bridge unavailable at {url}: {exc}") from exc
        decision = ManualDecision.from_payload(payload)
        self._last_error = None
        return decision
