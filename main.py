from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator

app = FastAPI(
    title="Universal Language to English Translator API",
    description="Convert any language into English using API",
    version="1.0.0"
)

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    original_text: str
    detected_language: str
    translated_text: str
    status: str


@app.get("/")
def home():
    return {
        "message": "Translator API is running successfully"
    }


def translate_to_english(text):
    try:
        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

        return {
            "original_text": text,
            "detected_language": "auto",
            "translated_text": translated,
            "status": "success"
        }

    except Exception as e:
        return {
            "original_text": text,
            "detected_language": "unknown",
            "translated_text": "",
            "status": f"failed: {str(e)}"
        }


@app.post("/translate", response_model=TranslationResponse)
def translate_api(request: TranslationRequest):
    return translate_to_english(request.text)