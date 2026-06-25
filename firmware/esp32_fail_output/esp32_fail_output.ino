/*
  ESP32 fail-output bridge for NeuroIris blower inspection.

  Wire the reject/PLC input through proper level shifting or isolation as needed.
  The firmware drives GPIO 4 (often labeled D4 on ESP32 dev boards) HIGH when
  the Python inspection app sends "FAIL" over USB serial. It drives the pin LOW
  for "PASS", "LOW", "STANDBY", or "RESET".
*/

#ifndef FAIL_OUTPUT_PIN
#define FAIL_OUTPUT_PIN 4
#endif

String commandBuffer;

void setFailOutput(bool failed) {
  digitalWrite(FAIL_OUTPUT_PIN, failed ? HIGH : LOW);
  Serial.println(failed ? "FAIL_OUTPUT=HIGH" : "FAIL_OUTPUT=LOW");
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "FAIL" || command == "HIGH") {
    setFailOutput(true);
  } else if (command == "PASS" || command == "LOW" || command == "STANDBY" || command == "RESET") {
    setFailOutput(false);
  } else if (command == "PING") {
    Serial.println("PONG");
  } else if (command.length() > 0) {
    Serial.print("UNKNOWN=");
    Serial.println(command);
  }
}

void setup() {
  pinMode(FAIL_OUTPUT_PIN, OUTPUT);
  digitalWrite(FAIL_OUTPUT_PIN, LOW);
  Serial.begin(115200);
  Serial.println("ESP32_FAIL_OUTPUT_READY GPIO=4 BAUD=115200");
}

void loop() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      handleCommand(commandBuffer);
      commandBuffer = "";
    } else {
      commandBuffer += c;
    }
  }
}
