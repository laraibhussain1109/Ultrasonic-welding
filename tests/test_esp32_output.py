import sys
from types import SimpleNamespace

from blower_inspection import esp32_output
from blower_inspection.esp32_output import ESP32FailOutput, ESP32OutputConfig


class FakeSerial:
    def __init__(self, responses=None):
        self.is_open = True
        self.writes = []
        self.closed = False
        self.responses = list(responses or [])
        self.reset_count = 0

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def readline(self):
        return self.responses.pop(0) if self.responses else b""

    def reset_input_buffer(self):
        self.reset_count += 1

    def close(self):
        self.closed = True
        self.is_open = False


def test_disabled_output_does_not_connect():
    output = ESP32FailOutput(ESP32OutputConfig(port="/dev/ttyUSB0", enabled=False, connect_settle_s=0, transport="serial"))

    assert output.set_fail(True) is False
    assert output.last_error == "ESP32 output disabled"


def test_set_fail_writes_expected_commands():
    output = ESP32FailOutput(ESP32OutputConfig(port="/dev/ttyUSB0", connect_settle_s=0, transport="serial"))
    fake = FakeSerial()
    output._serial = fake

    assert output.set_fail(True) is True
    assert output.set_fail(False) is True
    assert fake.writes == [b"FAIL\n", b"PASS\n"]


def test_connect_falls_back_from_wrong_preferred_port(monkeypatch):
    monkeypatch.setattr(esp32_output, "candidate_ports", lambda preferred_port=None: [preferred_port, "COM7"])
    output = ESP32FailOutput(ESP32OutputConfig(port="COM3", connect_settle_s=0, scan_all_ports=True, transport="serial"))
    opened_ports = []

    def fake_open_port(port):
        opened_ports.append(port)
        if port == "COM7":
            output._serial = FakeSerial()
            output.connected_port = port
            output.last_error = None
            return True
        output.last_error = f"{port}: not available"
        return False

    monkeypatch.setattr(output, "_open_port", fake_open_port)

    assert output.connect() is True
    assert opened_ports == ["COM3", "COM7"]
    assert output.connected_port == "COM7"


def test_missing_pyserial_reports_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "serial":
            raise ModuleNotFoundError("No module named 'serial'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    output = ESP32FailOutput(ESP32OutputConfig(port="COM3", connect_settle_s=0, scan_all_ports=False, transport="serial"))

    assert output.connect() is False
    assert "python -m pip install pyserial" in output.last_error


def test_handshake_skips_non_esp32_port_and_uses_firmware_port(monkeypatch):
    opened = {}

    def serial_factory(port, _baudrate, timeout):
        responses = [b"PONG\n"] if port == "COM7" else []
        serial = FakeSerial(responses=responses)
        serial.timeout = timeout
        opened[port] = serial
        return serial

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=serial_factory))
    monkeypatch.setattr(esp32_output, "candidate_ports", lambda preferred_port=None: ["COM1", "COM7"])
    monkeypatch.setattr(esp32_output, "DEFAULT_HANDSHAKE_TIMEOUT_S", 0.01)
    output = ESP32FailOutput(ESP32OutputConfig(port="COM1", connect_settle_s=0, scan_all_ports=True, transport="serial"))

    assert output.connect() is True
    assert opened["COM1"].closed is True
    assert opened["COM1"].writes == [b"PING\n"]
    assert opened["COM7"].writes == [b"PING\n"]
    assert output.connected_port == "COM7"


def test_handshake_can_be_disabled_for_custom_firmware(monkeypatch):
    fake = FakeSerial()
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=lambda *_args, **_kwargs: fake))
    output = ESP32FailOutput(
        ESP32OutputConfig(port="COM9", connect_settle_s=0, scan_all_ports=False, require_handshake=False, transport="serial")
    )

    assert output.connect() is True
    assert fake.writes == []
    assert output.connected_port == "COM9"


def test_wifi_connect_and_set_fail(monkeypatch):
    calls = []

    def fake_http_get(self, path):
        calls.append(path)
        if path == "/ping":
            return "PONG"
        if path == "/fail":
            return "FAIL_OUTPUT=ACTIVE GPIO=4 LEVEL=HIGH"
        return "OK"

    monkeypatch.setattr(ESP32FailOutput, "_http_get", fake_http_get)
    output = ESP32FailOutput(ESP32OutputConfig(transport="wifi", wifi_base_url="http://192.168.4.1"))

    assert output.set_fail(True) is True
    assert output.connected_port == "http://192.168.4.1"
    assert calls == ["/ping", "/fail"]


def test_wifi_rejects_failed_ping(monkeypatch):
    monkeypatch.setattr(ESP32FailOutput, "_http_get", lambda self, path: "NOT_PONG")
    output = ESP32FailOutput(ESP32OutputConfig(transport="wifi", wifi_base_url="http://192.168.4.1"))

    assert output.connect() is False
    assert "WiFi handshake failed" in output.last_error
