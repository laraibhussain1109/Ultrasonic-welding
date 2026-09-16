import urllib.error

import pytest

from blower_inspection.fail_output import ESP32FailOutputBridge, FailOutputConfig, ManualDecision


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _limit):
        return b"OK"


def bridge():
    return ESP32FailOutputBridge(FailOutputConfig(base_url="http://esp32.local", timeout_s=0.01))


def test_fail_output_calls_fail_and_pass_endpoints(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = bridge()
    try:
        assert output.send_result(True).result(timeout=1) == "OK"
        assert output.send_result(True) is None
        assert output.send_result(False).result(timeout=1) == "OK"
    finally:
        output.close()

    assert calls == [("http://esp32.local/fail", 0.01), ("http://esp32.local/pass", 0.01)]


def test_fail_output_retries_same_state_after_error(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.URLError("offline")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = bridge()
    try:
        with pytest.raises(RuntimeError):
            output.send_result(True).result(timeout=1)
        assert output.send_result(True).result(timeout=1) == "OK"
    finally:
        output.close()

    assert calls == ["http://esp32.local/fail", "http://esp32.local/fail"]


def test_poll_manual_fail_decision_with_sector(monkeypatch):
    class DecisionResponse(FakeResponse):
        def read(self, _limit):
            return b'{"sequence":7,"result":"FAIL","sector":12}'

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.headers.get("Accept"), timeout))
        return DecisionResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = bridge()
    try:
        decision = output.poll_manual_decision().result(timeout=1)
    finally:
        output.close()

    assert decision == ManualDecision(sequence=7, result="FAIL", sector=12)
    assert calls == [("http://esp32.local/api/decision", "application/json", 0.01)]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"sequence": 1, "result": "FAIL", "sector": 15}, "sector from 1 to 14"),
        ({"sequence": 1, "result": "MAYBE"}, "Unsupported manual result"),
    ],
)
def test_manual_decision_rejects_invalid_mobile_input(payload, message):
    with pytest.raises(ValueError, match=message):
        ManualDecision.from_payload(payload)
