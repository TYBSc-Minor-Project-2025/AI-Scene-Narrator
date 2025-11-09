import cv2
import pytesseract
import pyttsx3
from typing import List
from .config import TESSERACT_CMD, OCR_LANG, TTS_RATE

_tts_engine = None
_last_spoken = ""


def init_tts(rate: int = TTS_RATE):
    """Initialize text-to-speech engine."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', rate)
    return _tts_engine


def speak(text: str):
    """Speak text aloud with de-duplication."""
    global _last_spoken
    if not text or text.strip() == _last_spoken:
        return
    _last_spoken = text.strip()
    engine = init_tts()
    engine.say(text)
    engine.runAndWait()


def ocr_read(image) -> str:
    """Perform OCR on a BGR image."""
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb, lang=OCR_LANG)
    return text


def make_scene_sentence(detected: List[str], ocr_text: str, max_objects: int = 6) -> str:
    """Generate a simple description from detections and OCR."""
    cleaned = []
    for d in detected:
        if d not in cleaned:
            cleaned.append(d)
        if len(cleaned) >= max_objects:
            break

    obj_part = ""
    if cleaned:
        if len(cleaned) == 1:
            obj_part = f"There is a {cleaned[0]} in view."
        elif len(cleaned) == 2:
            obj_part = f"There are a {cleaned[0]} and a {cleaned[1]} in view."
        else:
            obj_part = "There are " + ", ".join(cleaned[:-1]) + f" and {cleaned[-1]} in view."

    text_part = ""
    if ocr_text and ocr_text.strip():
        cleaned_ocr = " ".join(ocr_text.split()).strip()
        if len(cleaned_ocr) > 100:
            cleaned_ocr = cleaned_ocr[:97] + "..."
        text_part = f" Text detected: {cleaned_ocr}."

    out = (obj_part + text_part).strip()
    return out if out else "I don't see anything recognizable right now."
