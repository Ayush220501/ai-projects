
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import tensorflow as tf
import numpy as np
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []

# 🔑 YOUR API KEYS (demo version)
USDA_API_KEY = "dkXt1JvOiDoM1NRHg7OPMQ2lJzSGzqvaWm9OYMIh"
GROQ_API_KEY = "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"


# ================= LOAD MODEL =================
@app.on_event("startup")
def load_model():
    global model, class_names

    model = tf.keras.models.load_model("/content/food_vision_fine_tuned_model.keras")


    import tensorflow_datasets as tfds
    class_names = tfds.builder("food101").info.features["label"].names

    print("✅ MODEL LOADED:", model is not None)


# ================= IMAGE PREPROCESS =================
def preprocess_image(image_bytes):
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

# ================= NUTRITION API =================
def get_nutrition(food_name):
    try:
        res = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={
                "api_key": USDA_API_KEY,
                "query": food_name
            },
            timeout=5
        )

        data = res.json()

        if "foods" in data and len(data["foods"]) > 0:
            return data["foods"][0]

    except:
        pass

    return None


# ================= AI INSIGHT =================
def generate_insight(food_name):
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{
                    "role": "user",
                    "content": f"Give one fun 25-word fact about {food_name.replace('_',' ')} with emoji."
                }],
                "max_tokens": 80
            },
            timeout=8
        )

        data = res.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

    except:
        pass

    return f"Enjoy your {food_name.replace('_',' ')} 🍽️"


# ================= MAIN API =================
@app.post("/analyze-photo")
async def analyze_photo(file: UploadFile = File(...)):

    if model is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    processed = preprocess_image(image_bytes)

    preds = model.predict(processed, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    food_name = class_names[idx] if idx < len(class_names) else "unknown"

    # nutrition
    nutrition_data = get_nutrition(food_name)
    nutrients = []

    if nutrition_data and "foodNutrients" in nutrition_data:
        for n in nutrition_data["foodNutrients"][:5]:
            nutrients.append({
                "name": n.get("nutrientName", ""),
                "value": n.get("value", 0),
                "unit": n.get("unitName", "")
            })

    # insight
    insight = generate_insight(food_name)

    return {
        "food": food_name.replace("_", " ").title(),
        "confidence": round(confidence * 100, 1),
        "nutrients": nutrients,
        "insight": insight
    }


# ================= FRONTEND =================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
