/*
  ESP32 fail-output bridge for NeuroIris blower inspection.

  Default operation is WiFi hotspot/AP mode:
    SSID:     NeuroIris-ESP32
    Password: neuroiris123
    URL:      http://192.168.4.1

  Connect the production PC to that hotspot (or configure station mode below),
  then the Python app can call /fail and /pass over HTTP. USB serial commands
  still work for bench testing from Arduino IDE Serial Monitor.

  If your relay board is active-low, change FAIL_ACTIVE_LEVEL to LOW. If your
  ESP32 board labels D4 differently, change FAIL_OUTPUT_PIN to the actual GPIO
  number wired to the relay input.
*/

#include <WiFi.h>
#include <WebServer.h>

#ifndef FAIL_OUTPUT_PIN
#define FAIL_OUTPUT_PIN 4
#endif

#ifndef FAIL_ACTIVE_LEVEL
#define FAIL_ACTIVE_LEVEL HIGH
#endif

#ifndef WIFI_AP_MODE
#define WIFI_AP_MODE 1
#endif

#ifndef WIFI_AP_SSID
#define WIFI_AP_SSID "NeuroIris-ESP32"
#endif

#ifndef WIFI_AP_PASSWORD
#define WIFI_AP_PASSWORD "neuroiris123"
#endif

#ifndef WIFI_STA_SSID
#define WIFI_STA_SSID ""
#endif

#ifndef WIFI_STA_PASSWORD
#define WIFI_STA_PASSWORD ""
#endif

const int FAIL_INACTIVE_LEVEL = (FAIL_ACTIVE_LEVEL == HIGH) ? LOW : HIGH;
WebServer server(80);
String commandBuffer;
bool failActive = false;

String statusText() {
  String status = "ESP32_FAIL_OUTPUT_READY";
  status += " GPIO=" + String(FAIL_OUTPUT_PIN);
  status += " ACTIVE_LEVEL=" + String(FAIL_ACTIVE_LEVEL == HIGH ? "HIGH" : "LOW");
  status += " STATE=" + String(failActive ? "FAIL" : "PASS");
  status += " IP=" + WiFi.localIP().toString();
  return status;
}

String setFailOutput(bool failed) {
  failActive = failed;
  digitalWrite(FAIL_OUTPUT_PIN, failed ? FAIL_ACTIVE_LEVEL : FAIL_INACTIVE_LEVEL);
  String response = "FAIL_OUTPUT=";
  response += failed ? "ACTIVE" : "INACTIVE";
  response += " GPIO=" + String(FAIL_OUTPUT_PIN);
  response += " LEVEL=" + String(digitalRead(FAIL_OUTPUT_PIN) == HIGH ? "HIGH" : "LOW");
  Serial.println(response);
  return response;
}

void sendText(int code, const String &body) {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(code, "text/plain", body);
}

void handleHttpRoutes() {
  server.on("/", HTTP_GET, []() { sendText(200, statusText()); });
  server.on("/ping", HTTP_GET, []() { sendText(200, "PONG"); });
  server.on("/status", HTTP_GET, []() { sendText(200, statusText()); });
  server.on("/fail", HTTP_GET, []() { sendText(200, setFailOutput(true)); });
  server.on("/pass", HTTP_GET, []() { sendText(200, setFailOutput(false)); });
  server.on("/high", HTTP_GET, []() { sendText(200, setFailOutput(true)); });
  server.on("/low", HTTP_GET, []() { sendText(200, setFailOutput(false)); });
  server.onNotFound([]() { sendText(404, "UNKNOWN_ENDPOINT"); });
  server.begin();
}

void startWiFi() {
#if WIFI_AP_MODE
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
  Serial.print("WIFI_AP_READY SSID=");
  Serial.print(WIFI_AP_SSID);
  Serial.print(" IP=");
  Serial.println(WiFi.softAPIP());
#else
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASSWORD);
  Serial.print("WIFI_STA_CONNECTING SSID=");
  Serial.println(WIFI_STA_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WIFI_STA_READY IP=");
  Serial.println(WiFi.localIP());
#endif
}

void printStatus() {
  Serial.println(statusText());
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
  startWiFi();
  handleHttpRoutes();
  printStatus();
}

void loop() {
  server.handleClient();
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
