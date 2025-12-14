import os
import time
import json
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import cv2

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import mediapipe as mp

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent.parent          # F:\HardwareChallenge
BACKEND_DIR = PROJECT_DIR / "backend"
UI_DIR = PROJECT_DIR / "ui"

PROMPT_PATH = BACKEND_DIR / "prompt.txt"
LOG_PATH = BACKEND_DIR / "logs.jsonl"
TMP_WAV = BACKEND_DIR / "tmp.wav"
BACKEND_DIR = Path(__file__).resolve().parent
YOLO_ONNX_PATH = BACKEND_DIR / "models" / "yolov5n.onnx"

# ENV 
load_dotenv(BACKEND_DIR / ".env")

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY", "") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL", "mistralai/mistral-7b-instruct:free") or "").strip()
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") or "").strip()

# CONFIG 
# Motion detection
MOTION_THRESHOLD = 25
MOTION_AREA_MIN = 2500
MOTION_IDLE_SEC = 8

# Audio record
REC_SECONDS = 5
SAMPLE_RATE = 16000

# Camera index (если нужно поменять, поставь 1 или 2)
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))

app = FastAPI()

cap = None  # глобальная камера
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))  # или поставь 0 вручную

# CORS (чтобы UI мог дергать API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

@app.get("/")
def serve_ui():
    return FileResponse(str(UI_DIR / "index.html"))

@app.get("/ping")
def ping():
    return {"ok": True, "ui_dir": str(UI_DIR), "backend_dir": str(BACKEND_DIR)}


dialog_history = []
state = {
    "mode": "idle",
    "last_motion_ts": 0,
    "last_person_ts": 0,
    "last_yolo_ts": 0,
    "greeted": False,
    "person_prev": False,
    "last_tts": ""
}

def load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты — виртуальный ресепшн. Задавай уточняющие вопросы и направляй клиента."

def log_event(event: dict):
    event["ts"] = time.time()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def record_wav(seconds=REC_SECONDS, sr=SAMPLE_RATE, out_path=TMP_WAV):
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype=np.int16)
    sd.wait()
    wav_write(str(out_path), sr, audio)
    return out_path

def yolo_person_present(frame_bgr, conf_th=0.35, iou_th=0.45, inp=640):
    img = cv2.resize(frame_bgr, (inp, inp))
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (inp, inp), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    out = yolo_net.forward()          # (1, 25200, 85)
    out = np.squeeze(out)             # (25200, 85)

    boxes, scores = [], []

    for det in out:
        cx, cy, w, h = det[0:4]
        obj = float(det[4])
        cls_probs = det[5:]
        cls_id = int(np.argmax(cls_probs))
        cls_conf = float(cls_probs[cls_id])
        conf = obj * cls_conf

        if cls_id != 0 or conf < conf_th:   # person=0
            continue

        x = cx - w/2
        y = cy - h/2
        boxes.append([x, y, w, h])
        scores.append(conf)

    if not boxes:
        return False, 0.0

    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_th, iou_th)
    if len(idxs) == 0:
        return False, 0.0

    best = max(scores[i] for i in idxs.flatten())
    return True, float(best)

# STT model (локально)
stt = WhisperModel("small", device="cpu", compute_type="int8")

yolo_net = cv2.dnn.readNetFromONNX(str(YOLO_ONNX_PATH))
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def stt_transcribe(wav_path: Path) -> str:
    segments, info = stt.transcribe(str(wav_path), language="ru")
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text

# LLM client (OpenAI-compatible API)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def clean_answer(text: str) -> str:
    if not text:
        return text
    for junk in ["<s>", "</s>", "[OUT]"]:
        text = text.replace(junk, "")
    return text.strip()

def llm_reply(user_text: str) -> str:
    prompt = load_prompt()

    if not OPENAI_API_KEY:
        return "ИИ не настроен: отсутствует API ключ."

    messages = [{"role": "system", "content": prompt}] + dialog_history + [{"role": "user", "content": user_text}]

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
        )
        answer = resp.choices[0].message.content.strip()
        answer = clean_answer(answer)
    except Exception as e:
        answer = f"ИИ временно недоступен. Ошибка: {type(e).__name__}"

    dialog_history.append({"role": "user", "content": user_text})
    dialog_history.append({"role": "assistant", "content": answer})
    return answer


@app.get("/status")
def get_status():
    return JSONResponse(state)

@app.post("/reset")
def reset_dialog():
    dialog_history.clear()
    state["mode"] = "idle"
    state["last_text"] = ""
    state["last_reply"] = ""
    log_event({"type": "reset"})
    return {"ok": True}

@app.post("/step")
def step_once():
    """
    Записать 5 сек голоса -> STT -> LLM -> вернуть текст.
    """
    state["mode"] = "listening"
    wav_path = record_wav()

    state["mode"] = "thinking"
    user_text = stt_transcribe(Path(wav_path))
    state["last_text"] = user_text

    reply = llm_reply(user_text if user_text else "Посетитель молчит. Вежливо попроси повторить.")
    state["last_reply"] = reply
    state["mode"] = "speaking"

    log_event({"type": "dialog_turn", "user": user_text, "assistant": reply})
    return {"user_text": user_text, "reply": reply}

@app.post("/motion_run")
def motion_run():
    global state, cap

    now = time.time()

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            return {"mode": "idle", "person": False, "say": "Не удалось открыть камеру."}

    ret, frame = cap.read()
    if not ret:
        return {"mode": state["mode"], "person": False, "say": ""}

    frame_small = cv2.resize(frame, (640, 360))

    # MOTION DETECTOR 
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    motion_detected = False
    ratio = 0.0

    prev_gray = state.get("prev_gray", None)
    if prev_gray is None:
        state["prev_gray"] = gray
    else:
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        ratio = float(np.sum(thresh > 0) / thresh.size)

        if ratio > 0.01:
            motion_detected = True
            state["last_motion_ts"] = now

        state["prev_gray"] = gray

    # YOLO (не каждый вызов) 
    yolo_now = False
    conf = 0.0

    if motion_detected or (now - state.get("last_yolo_ts", 0) > 0.7):
        yolo_now, conf = yolo_person_present(frame_small)
        state["last_yolo_ts"] = now
        if yolo_now:
            state["last_person_ts"] = now

    # стабилизация присутствия 
    PERSON_HOLD_SEC = 1.5
    person_stable = (now - state.get("last_person_ts", 0)) < PERSON_HOLD_SEC

    # режим 
    if person_stable:
        state["mode"] = "active"
    elif now - state.get("last_motion_ts", 0) > MOTION_IDLE_SEC:
        state["mode"] = "idle"

    # приветствие при появлении 
    if person_stable and not state.get("person_prev", False):
        if not state.get("greeted", False):
            state["last_tts"] = "Здравствуйте! Чем могу помочь?"
            state["greeted"] = True

    # ушёл сброс
    if not person_stable:
        state["greeted"] = False

    state["person_prev"] = person_stable

    # HUD (можно потом убрать) 
    cv2.putText(frame_small, f"mode: {state['mode']}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame_small, f"motion: {motion_detected}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame_small, f"yolo_now: {yolo_now} conf:{conf:.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame_small, f"person_stable: {person_stable}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame_small, f"ratio: {ratio:.3f}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    cv2.imshow("Receptionist Motion", frame_small)
    cv2.waitKey(1)

    # отдать фразу 1 раз и очистить
    out_tts = state.get("last_tts", "")
    state["last_tts"] = ""

    return {
        "mode": state["mode"],
        "person": person_stable,
        "say": out_tts
    }