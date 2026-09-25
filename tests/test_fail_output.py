import urllib.error

import pytest

from blower_inspection.fail_output import ESP32FailOutputBridge, FailOutputConfig


class FakeResponse:
    body = b"OK"

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _limit):
        return self.body


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


def test_each_final_pass_queues_a_pass_pulse(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = bridge()
    try:
        assert output.signal_pass().result(timeout=1) == "OK"
        assert output.signal_pass().result(timeout=1) == "OK"
    finally:
        output.close()

    assert calls == [
        ("http://esp32.local/pass-pulse", 0.01),
        ("http://esp32.local/pass-pulse", 0.01),
    ]


def test_poll_supervisor_decision_parses_sequenced_fail(monkeypatch):
    class DecisionResponse(FakeResponse):
        body = b'{"sequence":7,"result":"FAIL","sector":4}'

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: DecisionResponse())
    output = bridge()
    try:
        decision = output.poll_supervisor_decision().result(timeout=1)
    finally:
        output.close()

    assert decision.sequence == 7
    assert decision.result == "FAIL"
    assert decision.sector == 4


def test_poll_supervisor_decision_rejects_invalid_fail_sector(monkeypatch):
    class DecisionResponse(FakeResponse):
        body = b'{"sequence":2,"result":"FAIL","sector":15}'

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: DecisionResponse())
    output = bridge()
    try:
        with pytest.raises(RuntimeError, match="FAIL sector"):
            output.poll_supervisor_decision().result(timeout=1)
    finally:
        output.close()
