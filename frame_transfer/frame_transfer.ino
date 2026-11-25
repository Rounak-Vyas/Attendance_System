#include "esp_camera.h"
#include "img_converters.h"
#include "esp_log.h"

// LED Pins
#define GREEN_LED 1
#define RED_LED   2

// OV7670 Pin Mapping (PCLK updated to GPIO 18)
#define CAM_PIN_PWDN  -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK  15
#define CAM_PIN_SIOD  4
#define CAM_PIN_SIOC  5

#define CAM_PIN_D7    17
#define CAM_PIN_D6    14
#define CAM_PIN_D5    13
#define CAM_PIN_D4    12
#define CAM_PIN_D3    11
#define CAM_PIN_D2    10
#define CAM_PIN_D1    9
#define CAM_PIN_D0    8

#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF  7
#define CAM_PIN_PCLK  18

// 4-byte sync header so Python always finds frame boundaries
const uint8_t SYNC_HEADER[4] = {0xAA, 0x55, 0xAA, 0x55};

// LED state machine
char ledMode = 'O';            // 'G' = blink green, 'R' = solid red, 'O' = off
uint32_t lastCmdTime = 0;
uint32_t lastBlinkTime = 0;
bool greenState = false;

void updateLEDs() {
  uint32_t now = millis();

  // Read any incoming command from PC
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'G' || cmd == 'R' || cmd == 'O') {
      ledMode = cmd;
      lastCmdTime = now;
    }
  }

  // Safety timeout: turn LEDs off if PC stops sending updates for 2s
  if (now - lastCmdTime > 2000) {
    ledMode = 'O';
  }

  if (ledMode == 'G') {
    digitalWrite(RED_LED, LOW);
    if (now - lastBlinkTime >= 150) {
      lastBlinkTime = now;
      greenState = !greenState;
      digitalWrite(GREEN_LED, greenState ? HIGH : LOW);
    }
  } else if (ledMode == 'R') {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);
  } else {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, LOW);
  }
}

void setup() {
  esp_log_level_set("*", ESP_LOG_NONE); // Keep USB binary stream clean
  Serial.begin(115200);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = CAM_PIN_D0;
  config.pin_d1       = CAM_PIN_D1;
  config.pin_d2       = CAM_PIN_D2;
  config.pin_d3       = CAM_PIN_D3;
  config.pin_d4       = CAM_PIN_D4;
  config.pin_d5       = CAM_PIN_D5;
  config.pin_d6       = CAM_PIN_D6;
  config.pin_d7       = CAM_PIN_D7;
  config.pin_xclk     = CAM_PIN_XCLK;
  config.pin_pclk     = CAM_PIN_PCLK;
  config.pin_vsync    = CAM_PIN_VSYNC;
  config.pin_href     = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn     = CAM_PIN_PWDN;
  config.pin_reset    = CAM_PIN_RESET;

  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_QVGA;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) != ESP_OK) {
    digitalWrite(RED_LED, HIGH); // Solid red indicates init failure
    while (true) delay(1000);
  }
}

void loop() {
  updateLEDs();

  // Only stream when a PC app has opened the USB CDC port
  if (!Serial) {
    delay(50);
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  uint8_t *jpg_buf = NULL;
  size_t jpg_len = 0;
  bool ok = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, fb->format, 75, &jpg_buf, &jpg_len);
  esp_camera_fb_return(fb);

  if (ok && jpg_len > 0) {
    uint32_t len32 = (uint32_t)jpg_len;
    Serial.write(SYNC_HEADER, 4);
    Serial.write((uint8_t *)&len32, 4);
    Serial.write(jpg_buf, jpg_len);
    free(jpg_buf);
  }
}