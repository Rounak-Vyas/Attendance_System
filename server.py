import asyncio
import base64
import struct
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import serial
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis
from pydantic import BaseModel

import database

PORT = "COM11"          # <-- Change to your ESP32-S3 COM port
MATCH_THRESHOLD = 0.45
SYNC = b"\xAA\x55\xAA\x55"

state_lock = threading.Lock()
stop_event = threading.Event()
new_frame_event = threading.Event()

shared = {
    "mode": "detection",
    "led_cmd": b"O",
    "latest_raw_jpg": None,
    "frame_b64": None,
    "frame_id": 0,
    "state_version": 0,
    "latest_emb": None,
    "known_users": [],
    "cached_users": [],
    "cached_logs": [],
    "metrics": {"fps": 0.0, "infer_ms": 0, "provider": "Loading..."},
    "status": {
        "state": "idle",
        "led": "O",
        "face_detected": False,
        "name": None,
        "user_code": None,
        "confidence": 0.0,
    },
}


def serial_reader_worker():
    """Continuously drains USB CDC and keeps ONLY the single newest frame."""
    print(f"Opening serial port {PORT}...")
    try:
        ser = serial.Serial(PORT, 115200, timeout=0.05)
    except Exception as e:
        print(f"[ERROR] Could not open {PORT}: {e}")
        return

    ser.reset_input_buffer()
    buf = bytearray()
    last_sent_cmd = None
    last_heartbeat = 0.0

    try:
        while not stop_event.is_set():
            now = time.time()
            with state_lock:
                cmd = shared["led_cmd"]

            # Send LED command immediately on change, plus 0.4s heartbeat
            if cmd != last_sent_cmd or (now - last_heartbeat >= 0.4):
                ser.write(cmd)
                last_sent_cmd = cmd
                last_heartbeat = now

            chunk = ser.read(ser.in_waiting or 4096)
            if not chunk:
                continue
            buf.extend(chunk)

            # Parse every complete frame in buffer, keep only the LAST (newest) one
            newest_jpg = None
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

                newest_jpg = bytes(buf[8 : 8 + frame_len])
                del buf[: 8 + frame_len]

            if newest_jpg is not None:
                with state_lock:
                    shared["latest_raw_jpg"] = newest_jpg
                new_frame_event.set()
    finally:
        try:
            ser.write(b"O")
            ser.close()
        except Exception:
            pass


def ai_inference_worker():
    """Processes only the newest available frame so latency never builds up."""
    database.init_db()
    with state_lock:
        shared["known_users"] = database.get_all_users(include_embeddings=True)
        shared["cached_users"] = database.get_all_users(include_embeddings=False)
        shared["cached_logs"] = database.get_recent_logs(15)

    print("Loading InsightFace (detection + recognition only)...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    active_provider = "CPU"
    try:
        for model in face_app.models.values():
            if hasattr(model, "session") and "CUDAExecutionProvider" in model.session.get_providers():
                active_provider = "CUDA GPU"
                break
    except Exception:
        pass
    print(f"InsightFace ready on {active_provider}!")

    with state_lock:
        shared["metrics"]["provider"] = active_provider
        shared["state_version"] += 1

    prev_frame_time = None
    smooth_fps = 0.0

    while not stop_event.is_set():
        if not new_frame_event.wait(timeout=0.5):
            continue
        new_frame_event.clear()

        with state_lock:
            raw_jpg = shared["latest_raw_jpg"]
            mode = shared["mode"]
            known_users = list(shared["known_users"])

        if raw_jpg is None:
            continue

        frame = cv2.imdecode(np.frombuffer(raw_jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        t0 = time.perf_counter()
        faces = face_app.get(frame)
        infer_ms = int((time.perf_counter() - t0) * 1000)

        display = cv2.resize(frame, (640, 480))
        new_status = {
            "state": "idle",
            "led": "O",
            "face_detected": False,
            "name": None,
            "user_code": None,
            "confidence": 0.0,
        }
        latest_emb = None
        led_cmd = b"O"
        logs_updated = False

        if len(faces) > 0:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.embedding / np.linalg.norm(face.embedding)
            latest_emb = emb
            new_status["face_detected"] = True

            x1, y1, x2, y2 = [int(v * 2) for v in face.bbox]

            if mode == "enrollment":
                led_cmd = b"O"
                new_status["state"] = "ready_to_enroll"
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(
                    display, "Ready to Enroll", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2,
                )
            else:
                best_user = None
                best_score = -1.0
                for user in known_users:
                    score = float(np.dot(emb, user["embedding"]))
                    if score > best_score:
                        best_score = score
                        best_user = user

                if best_user and best_score >= MATCH_THRESHOLD:
                    led_cmd = b"G"
                    color = (0, 255, 0)
                    label = f"{best_user['name']} ({best_score:.2f})"
                    new_status.update({
                        "state": "matched",
                        "led": "G",
                        "name": best_user["name"],
                        "user_code": best_user["user_code"],
                        "confidence": round(best_score, 3),
                    })
                    if database.log_attendance(
                        best_user["id"], best_user["name"], best_user["user_code"], best_score
                    ):
                        logs_updated = True
                else:
                    led_cmd = b"R"
                    color = (0, 0, 255)
                    label = f"Unknown ({best_score:.2f})" if best_score > 0 else "Unknown"
                    new_status.update({
                        "state": "unknown",
                        "led": "R",
                        "confidence": round(max(0.0, best_score), 3),
                    })

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )

        now = time.time()
        if prev_frame_time is not None:
            dt = now - prev_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                smooth_fps = inst_fps if smooth_fps == 0 else (0.7 * smooth_fps + 0.3 * inst_fps)
        prev_frame_time = now

        ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            frame_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            new_logs = database.get_recent_logs(15) if logs_updated else None

            with state_lock:
                shared["led_cmd"] = led_cmd
                shared["frame_b64"] = frame_b64
                shared["latest_emb"] = latest_emb
                shared["status"] = new_status
                shared["metrics"] = {
                    "fps": round(smooth_fps, 1),
                    "infer_ms": infer_ms,
                    "provider": active_provider,
                }
                if new_logs is not None:
                    shared["cached_logs"] = new_logs
                shared["frame_id"] += 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    t_serial = threading.Thread(target=serial_reader_worker, daemon=True)
    t_ai = threading.Thread(target=ai_inference_worker, daemon=True)
    t_serial.start()
    t_ai.start()
    yield
    stop_event.set()
    new_frame_event.set()
    t_serial.join(timeout=2)
    t_ai.join(timeout=2)


app = FastAPI(lifespan=lifespan)


class ModeRequest(BaseModel):
    mode: str


class EnrollRequest(BaseModel):
    name: str
    user_code: str


@app.get("/")
def serve_dashboard():
    return FileResponse("index.html")


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    if req.mode not in ("detection", "enrollment"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    with state_lock:
        shared["mode"] = req.mode
        if req.mode == "enrollment":
            shared["led_cmd"] = b"O"
        shared["state_version"] += 1
    return {"mode": req.mode}


@app.post("/api/enroll")
def enroll_user(req: EnrollRequest):
    name = req.name.strip()
    code = req.user_code.strip()
    if not name or not code:
        raise HTTPException(status_code=400, detail="Name and ID are required")

    with state_lock:
        emb = shared["latest_emb"]

    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected in frame. Look at the camera and try again.")

    database.add_user(name, code, emb)
    known = database.get_all_users(include_embeddings=True)
    cached = database.get_all_users(include_embeddings=False)
    with state_lock:
        shared["known_users"] = known
        shared["cached_users"] = cached
        shared["mode"] = "detection"
        shared["state_version"] += 1

    return {"message": f"Enrolled {name} ({code})"}


@app.delete("/api/users/{user_id}")
def remove_user(user_id: int):
    database.delete_user(user_id)
    known = database.get_all_users(include_embeddings=True)
    cached = database.get_all_users(include_embeddings=False)
    with state_lock:
        shared["known_users"] = known
        shared["cached_users"] = cached
        shared["state_version"] += 1
    return {"deleted": user_id}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    last_frame_id = -1
    last_state_ver = -1
    try:
        while not stop_event.is_set():
            payload = None
            with state_lock:
                fid = shared["frame_id"]
                sver = shared["state_version"]
                if fid != last_frame_id or sver != last_state_ver:
                    last_frame_id = fid
                    last_state_ver = sver
                    payload = {
                        "frame": shared["frame_b64"],
                        "mode": shared["mode"],
                        "status": dict(shared["status"]),
                        "metrics": dict(shared["metrics"]),
                        "users": list(shared["cached_users"]),
                        "logs": list(shared["cached_logs"]),
                    }
            if payload is not None:
                await ws.send_json(payload)
            await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        pass