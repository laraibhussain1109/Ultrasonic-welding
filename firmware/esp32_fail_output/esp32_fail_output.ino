/*
  ESP32 fail-output bridge for NeuroIris blower inspection.

  Default operation is WiFi hotspot/AP mode:
    SSID:     NeuroIris-ESP32
    Password: neuroiris123
    URL:      http://192.168.4.1

  Connect the production PC to that hotspot (or configure station mode below),
  then open http://192.168.4.1 on a phone to enter an operator decision. The
  Python app polls /api/decision and draws a selected failed sector on camera.
  USB serial commands still work for bench testing from Arduino IDE Serial Monitor.

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
String manualResult = "RECHECK";
int manualSector = 0;
uint32_t decisionSequence = 0;

const char OPERATOR_PAGE[] PROGMEM = R"HTML(
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroIris Manual Inspection</title><style>
body{margin:0;background:#07111f;color:#d9eaff;font:700 18px Arial;text-align:center}main{max-width:560px;margin:auto;padding:24px}
h1{color:#20dfff;font-size:25px}.button{display:block;width:100%;box-sizing:border-box;border:0;border-radius:12px;padding:20px;margin:14px 0;font-size:24px;font-weight:800;color:white}
.pass{background:#08783f}.fail{background:#b51627}.recheck{background:#a66500}.panel{display:none;background:#111f32;border-radius:12px;padding:16px}.sectors{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.sector{padding:15px 4px;background:#263b57;border:2px solid #57799e;border-radius:8px;color:white;font-size:20px}.sector.selected{background:#c51f31;border-color:#ff8290}#status{min-height:24px;color:#75ffba}</style></head>
<body><main><h1>MANUAL INSPECTION</h1><p>Select the result for the part on camera.</p>
<button class="button pass" onclick="sendDecision('PASS')">PASS</button>
<button class="button fail" onclick="showSectors()">FAIL</button>
<section class="panel" id="failPanel"><p>Select failed sector (1–14)</p><div class="sectors" id="sectors"></div></section>
<button class="button recheck" onclick="sendDecision('RECHECK')">RECHECK</button><p id="status">Ready</p></main>
<script>const box=document.getElementById('sectors');for(let n=1;n<=14;n++){let b=document.createElement('button');b.className='sector';b.textContent=n;b.onclick=()=>sendDecision('FAIL',n);box.appendChild(b)}
function showSectors(){document.getElementById('failPanel').style.display='block'}
async function sendDecision(result,sector){let u='/api/decision?result='+result+(sector?'&sector='+sector:'');try{let r=await fetch(u,{method:'POST'});let d=await r.json();document.getElementById('status').textContent='Sent: '+d.result+(d.sector?' — sector '+d.sector:'');document.querySelectorAll('.sector').forEach((b,i)=>b.classList.toggle('selected',i+1===d.sector));if(result!=='FAIL')document.getElementById('failPanel').style.display='none'}catch(e){document.getElementById('status').textContent='Could not send — retry'}}
</script></body></html>)HTML";

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

String decisionJson() {
  String json = "{\"sequence\":" + String(decisionSequence) + ",\"result\":\"" + manualResult + "\",\"sector\":";
  json += manualSector > 0 ? String(manualSector) : "null";
  return json + "}";
}

void sendDecisionJson(int code) {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(code, "application/json", decisionJson());
}

void handleManualDecision() {
  String result = server.arg("result");
  result.toUpperCase();
  int sector = server.arg("sector").toInt();
  if (result != "PASS" && result != "FAIL" && result != "RECHECK") {
    sendText(400, "RESULT_MUST_BE_PASS_FAIL_OR_RECHECK");
    return;
  }
  if (result == "FAIL" && (sector < 1 || sector > 14)) {
    sendText(400, "FAIL_SECTOR_MUST_BE_1_TO_14");
    return;
  }
  manualResult = result;
  manualSector = result == "FAIL" ? sector : 0;
  decisionSequence++;
  setFailOutput(result == "FAIL");
  Serial.println("MANUAL_DECISION=" + manualResult + " SECTOR=" + String(manualSector) + " SEQUENCE=" + String(decisionSequence));
  sendDecisionJson(200);
}

void handleHttpRoutes() {
  server.on("/", HTTP_GET, []() { server.send_P(200, "text/html", OPERATOR_PAGE); });
  server.on("/ping", HTTP_GET, []() { sendText(200, "PONG"); });
  server.on("/status", HTTP_GET, []() { sendText(200, statusText()); });
  server.on("/api/decision", HTTP_GET, []() { sendDecisionJson(200); });
  server.on("/api/decision", HTTP_POST, handleManualDecision);
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
