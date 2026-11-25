import serial
import struct
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import database

PORT = "COM11"         # <-- Change to your ESP32-S3 COM port
MATCH_THRESHOLD = 0.45
SYNC = b"\xAA\x55\xAA\x55"

database.init_db()
known_users = database.get_all_users()

print("Loading InsightFace model (first run downloads ~280 MB)...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))
print("Model loaded!")

ser = serial.Serial(PORT, 115200, timeout=0.05)
buf = bytearray()
led_cmd = b"O"
last_heartbeat = time.time()
latest_face_emb = None

print("Running! Keys: [e] Enroll current face | [q] Quit")

try:
    while True:
        now = time.time()
        if now - last_heartbeat >= 0.5:
            ser.write(led_cmd)
            last_heartbeat = now

        chunk = ser.read(ser.in_waiting or 4096)
        if chunk:
            buf.extend(chunk)

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
            if frame is None:
                continue

            frame_decoded = True
            faces = app.get(frame)
            display = cv2.resize(frame, (640, 480))
            scale_x, scale_y = 2.0, 2.0

            if len(faces) == 0:
                led_cmd = b"O"
                latest_face_emb = None
            else:
                # Pick the largest face in the frame
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                emb = face.embedding / np.linalg.norm(face.embedding)
                latest_face_emb = emb

                best_user = None
                best_score = -1.0
                for user in known_users:
                    score = float(np.dot(emb, user["embedding"]))
                    if score > best_score:
                        best_score = score
                        best_user = user

                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                if best_user and best_score >= MATCH_THRESHOLD:
                    led_cmd = b"G"
                    color = (0, 255, 0)
                    label = f"{best_user['name']} ({best_score:.2f})"
                    if database.log_attendance(best_user["id"], best_user["name"], best_user["user_code"], best_score):
                        print(f"[ATTENDANCE LOGGED] {best_user['name']} ({best_user['user_code']}) | Score: {best_score:.2f}")
                else:
                    led_cmd = b"R"
                    color = (0, 0, 255)
                    label = f"Unknown ({best_score:.2f})" if best_score > 0 else "Unknown"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            ser.write(led_cmd)
            cv2.putText(display, f"LED: {led_cmd.decode()} | Press [e] to Enroll", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Smart Attendance - AI Test", display)

        if frame_decoded or not chunk:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                if latest_face_emb is None:
                    print("No face detected right now. Look at the camera and press [e] again.")
                else:
                    name = input("\nEnter User Name: ").strip()
                    code = input("Enter ID / Roll No: ").strip()
                    if name and code:
                        database.add_user(name, code, latest_face_emb)
                        known_users = database.get_all_users()
                        print(f"Enrolled {name} ({code})! Total users: {len(known_users)}\n")
finally:
    ser.write(b"O")
    ser.close()
    cv2.destroyAllWindows()