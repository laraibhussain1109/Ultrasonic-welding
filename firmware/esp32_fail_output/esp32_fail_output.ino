/*
  ESP32 fail-output bridge for NeuroIris blower inspection.

  Wire the reject/PLC input through proper level shifting or isolation as needed.
  The firmware drives GPIO 4 (often labeled D4 on ESP32 dev boards) HIGH when
  the Python inspection app sends "FAIL" over USB serial. It drives the pin LOW
  for "PASS", "LOW", "STANDBY", or "RESET".

  If your relay board is active-low, change FAIL_ACTIVE_LEVEL to LOW. If your
  ESP32 board labels D4 differently, change FAIL_OUTPUT_PIN to the actual GPIO
  number wired to the relay input.
*/

#ifndef FAIL_OUTPUT_PIN
#define FAIL_OUTPUT_PIN 4
#endif

#ifndef FAIL_ACTIVE_LEVEL
#define FAIL_ACTIVE_LEVEL HIGH
#endif

const int FAIL_INACTIVE_LEVEL = (FAIL_ACTIVE_LEVEL == HIGH) ? LOW : HIGH;
String commandBuffer;
bool failActive = false;

void setFailOutput(bool failed) {
  failActive = failed;
  digitalWrite(FAIL_OUTPUT_PIN, failed ? FAIL_ACTIVE_LEVEL : FAIL_INACTIVE_LEVEL);
  Serial.print("FAIL_OUTPUT=");
  Serial.print(failed ? "ACTIVE" : "INACTIVE");
  Serial.print(" GPIO=");
  Serial.print(FAIL_OUTPUT_PIN);
  Serial.print(" LEVEL=");
  Serial.println(digitalRead(FAIL_OUTPUT_PIN) == HIGH ? "HIGH" : "LOW");
}

void printStatus() {
  Serial.print("ESP32_FAIL_OUTPUT_READY GPIO=");
  Serial.print(FAIL_OUTPUT_PIN);
  Serial.print(" ACTIVE_LEVEL=");
  Serial.print(FAIL_ACTIVE_LEVEL == HIGH ? "HIGH" : "LOW");
  Serial.print(" STATE=");
  Serial.println(failActive ? "FAIL" : "PASS");
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
  } else if (command == "STATUS") {
    printStatus();
  } else if (command.length() > 0) {
    Serial.print("UNKNOWN=");
    Serial.println(command);
  }
}

void setup() {
  pinMode(FAIL_OUTPUT_PIN, OUTPUT);
  digitalWrite(FAIL_OUTPUT_PIN, FAIL_INACTIVE_LEVEL);
  Serial.begin(115200);
  delay(250);
  printStatus();
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
