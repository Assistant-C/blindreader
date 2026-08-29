from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from google.cloud import vision
from openai import OpenAI
from gtts import gTTS
import io
import os
from PIL import Image
import uuid

app = FastAPI()

# Завантаження API-ключів із середовища (НЕ пиши прямо в код!)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY не встановлено в середовищі")

openai_client = OpenAI(api_key=openai_api_key)
vision_client = vision.ImageAnnotatorClient()


@app.post("/recognize_with_gpt/")
async def recognize_with_gpt(file: UploadFile = File(...)):
    try:
        # Зчитування фото
        image_bytes = await file.read()
        image = vision.Image(content=image_bytes)

        # Vision API: Отримуємо весь текст
        response = vision_client.document_text_detection(image=image)
        full_text = response.full_text_annotation.text

        # Промпт до GPT
        prompt = f"""
        Ти — асистент, який допомагає зчитати накладну.
        Ось текст з неї:
        {full_text}
        Витягни назви товарів і кількість у форматі звичайного тексту по рядках.
        Приклад:
        Кефір Славія 2,5% — 3 шт
        Йогурт полуниця — 2 шт
        Пиши лише список, без заголовків, без коментарів, без JSON.
        """

        gpt_response = openai_client.chat.completions.create( model="gpt-4-turbo", messages=[ {"role": "system", "content": "Ти досвідчений парсер накладних."}, {"role": "user", "content": prompt} ], temperature=0.1, max_tokens=2000 )

        result_text = gpt_response.choices[0].message.content.strip()

        # Обробка "штей"
        def normalize_units(text):
            return (
                text.replace(" 1 шт", " 1 штука")
                    .replace(" 2 шт", " 2 штуки")
                    .replace(" 3 шт", " 3 штуки")
                    .replace(" 4 шт", " 4 штуки")
                    .replace(" 5 шт", " 5 штук")
                    .replace(" 6 шт", " 6 штук")
                    .replace(" 7 шт", " 7 штук")
                    .replace(" 8 шт", " 8 штук")
                    .replace(" 9 шт", " 9 штук")
                    .replace(" 0 шт", " 0 штук")
                    .replace(" шт", " штук")
            )

        result_text = normalize_units(result_text)

        # Генерація аудіо
        os.makedirs("audio", exist_ok=True)
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("audio", filename)

        tts = gTTS(text=result_text, lang='uk')
        tts.save(filepath)

        return FileResponse(filepath, media_type="audio/mpeg", filename=filename)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/tts/")
async def generate_tts(text: str = Form(...)):
    try:
        text = text.strip()
        if not text:
            return JSONResponse(content={"error": "Порожній текст"}, status_code=400)

        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("audio", filename)
        os.makedirs("audio", exist_ok=True)

        tts = gTTS(text=text, lang='uk')
        tts.save(filepath)

        return {"url": f"/audio/{filename}"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    path = os.path.join("audio", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg", filename=filename)
    return JSONResponse(content={"error": "Файл не знайдено"}, status_code=404)
