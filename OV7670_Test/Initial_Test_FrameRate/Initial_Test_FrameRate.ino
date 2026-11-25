#include "esp_camera.h"
#include "img_converters.h"

// LED Pins
#define GREEN_LED 1
#define RED_LED   2

// OV7670 Pin Mapping
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

void setup() {
  Serial.begin(115200);
  delay(2000); // Wait for Native USB CDC to enumerate

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  // Quick LED hardware check
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, HIGH);
  delay(500);
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

  config.xclk_freq_hz = 10000000;          // 10 MHz for stable breadboard bring-up
  config.pixel_format = PIXFORMAT_RGB565;  // OV7670 outputs raw pixels, not JPEG
  config.frame_size   = FRAMESIZE_QVGA;    // 320x240
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init FAILED with error 0x%x\n", err);
    digitalWrite(RED_LED, HIGH);
    while (true) delay(1000);
  }

  Serial.println("OV7670 Initialized Successfully!");
}

void loop() {
  uint32_t t0 = millis();
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed!");
    delay(500);
    return;
  }

  uint8_t *jpg_buf = NULL;
  size_t jpg_len = 0;
  bool converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, fb->format, 80, &jpg_buf, &jpg_len);
  uint32_t dt = millis() - t0;

  if (converted) {
    Serial.printf("Raw: %u bytes | JPEG: %u bytes | Time: %lu ms\n", fb->len, jpg_len, dt);
    free(jpg_buf);
  } else {
    Serial.println("JPEG conversion failed!");
  }

  esp_camera_fb_return(fb);
  delay(1000);
}