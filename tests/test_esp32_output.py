from blower_inspection import esp32_output
from blower_inspection.esp32_output import ESP32FailOutput, ESP32OutputConfig


class FakeSerial:
    def __init__(self):
        self.is_open = True
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True
        self.is_open = False


def test_disabled_output_does_not_connect():
    output = ESP32FailOutput(ESP32OutputConfig(port="/dev/ttyUSB0", enabled=False, connect_settle_s=0))

    assert output.set_fail(True) is False
    assert output.last_error == "ESP32 output disabled"


def test_set_fail_writes_expected_commands():
    output = ESP32FailOutput(ESP32OutputConfig(port="/dev/ttyUSB0", connect_settle_s=0))
    fake = FakeSerial()
    output._serial = fake

    assert output.set_fail(True) is True
    assert output.set_fail(False) is True
    assert fake.writes == [b"FAIL\n", b"PASS\n"]


def test_connect_falls_back_from_wrong_preferred_port(monkeypatch):
    monkeypatch.setattr(esp32_output, "candidate_ports", lambda preferred_port=None: [preferred_port, "COM7"])
    output = ESP32FailOutput(ESP32OutputConfig(port="COM3", connect_settle_s=0, scan_all_ports=True))
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
