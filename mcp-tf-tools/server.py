import tensorflow as tf
from fastapi import FastAPI
import uvicorn

app = FastAPI()

model = tf.keras.models.load_model("sentiment_model.h5")

@app.post("/sentiment")
def sentiment(data: dict):
    text = data["text"]
    pred = float(model.predict([text])[0])
    return {"sentiment": pred}

@app.post("/add")
def add(data: dict):
    return {"result": data["a"] + data["b"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
