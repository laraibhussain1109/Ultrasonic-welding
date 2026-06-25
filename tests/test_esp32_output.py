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
    output = ESP32FailOutput(ESP32OutputConfig(port="/dev/ttyUSB0", enabled=False))

    assert output.set_fail(True) is False
    assert output.last_error == "ESP32 output disabled"


def test_set_fail_writes_expected_commands():
    output = ESP32FailOutput(ESP32OutputConfig(port="/dev/ttyUSB0"))
    fake = FakeSerial()
    output._serial = fake

    assert output.set_fail(True) is True
    assert output.set_fail(False) is True
    assert fake.writes == [b"FAIL\n", b"PASS\n"]
