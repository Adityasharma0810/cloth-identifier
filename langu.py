# main.py
# Universal Language to English Translator API
# Ready for Render Deployment

# Install dependencies:
# pip install fastapi uvicorn googletrans==4.0.0-rc1

from fastapi import FastAPI
from pydantic import BaseModel
from googletrans import Translator

app = FastAPI(
    title="Universal Language to English Translator API",
    description="Convert any language into English using API",
    version="1.0.0"
)

# Request model
class TranslationRequest(BaseModel):
    text: str

# Response model
class TranslationResponse(BaseModel):
    original_text: str
    detected_language: str
    translated_text: str
    status: str

# Home route
@app.get("/")
def home():
    return {
        "message": "Translator API is running successfully"
    }

# Translate function
def translate_to_english(text):
    translator = Translator()

    try:
        # Detect source language
        detected = translator.detect(text)

        # Translate to English
        translated = translator.translate(text, dest="en")

        return {
            "original_text": text,
            "detected_language": detected.lang,
            "translated_text": translated.text,
            "status": "success"
        }

    except Exception as e:
        return {
            "original_text": text,
            "detected_language": "unknown",
            "translated_text": "",
            "status": f"failed: {str(e)}"
        }

# API route
@app.post("/translate", response_model=TranslationResponse)
def translate_api(request: TranslationRequest):
    return translate_to_english(request.text)