import base64

# sentiment_model_base64.txt to sentiment_model.h5
with open("sentiment_model_base64.txt", "rb") as f:
    b64 = f.read()

binary = base64.b64decode(b64)

with open("sentiment_model.h5", "wb") as f:
    f.write(binary)

print("sentiment_model.h5 created successfully!")
