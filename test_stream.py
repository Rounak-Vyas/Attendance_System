import serial
import struct
import time
import cv2
import numpy as np

PORT = "COM11"  # <-- Keep your ESP32-S3 COM port here
SYNC = b"\xAA\x55\xAA\x55"

ser = serial.Serial(PORT, 115200, timeout=0.05)
buf = bytearray()

led_cmd = b"O"
last_heartbeat = time.time()
fps_count = 0
bytes_count = 0
fps_timer = time.time()
current_fps = 0.0
current_kbps = 0.0

print("Connected! Keys: [g] Green Blink | [r] Red Solid | [o] Off | [q] Quit")

try:
    while True:
        now = time.time()
        if now - last_heartbeat >= 0.5:
            ser.write(led_cmd)
            last_heartbeat = now

        # Grab all buffered bytes at once
        chunk = ser.read(ser.in_waiting or 4096)
        if chunk:
            buf.extend(chunk)
            bytes_count += len(chunk)

        frame_decoded = False
        while True:
            idx = buf.find(SYNC)
            if idx == -1:
                if len(buf) > 3:
                    del buf[:-3]
                break

            if idx > 0:
                del buf[:idx]

            if len(buf) < 8:
                break

            frame_len = struct.unpack("<I", buf[4:8])[0]
            if not (0 < frame_len < 100000):
                del buf[:4]
                continue

            if len(buf) < 8 + frame_len:
                break

            jpg_bytes = bytes(buf[8 : 8 + frame_len])
            del buf[: 8 + frame_len]

            frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_decoded = True
                fps_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    elapsed = now - fps_timer
                    current_fps = fps_count / elapsed
                    current_kbps = (bytes_count / 1024.0) / elapsed
                    fps_count = 0
                    bytes_count = 0
                    fps_timer = now

                display = cv2.resize(frame, (640, 480))
                cv2.putText(
                    display,
                    f"FPS: {current_fps:.1f} | {current_kbps:.0f} KB/s | LED: {led_cmd.decode()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("OV7670 Live Stream", display)

        # Only pause for GUI events after a full frame arrives
        if frame_decoded or not chunk:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                led_cmd = b"G"
                ser.write(led_cmd)
            elif key == ord("r"):
                led_cmd = b"R"
                ser.write(led_cmd)
            elif key == ord("o"):
                led_cmd = b"O"
                ser.write(led_cmd)
finally:
    ser.write(b"O")
    ser.close()
    cv2.destroyAllWindows()