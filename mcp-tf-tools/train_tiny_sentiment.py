import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

# train sentiment_model.h5 on tiny dataset and save it. This is just for demonstration - in real use case, you'd train on a larger dataset.

# Tiny dataset
texts = [
    "I love this", "This is great", "Amazing experience", "I feel happy",
    "I hate this", "This is bad", "Terrible experience", "I feel sad"
]

labels = [1, 1, 1, 1, 0, 0, 0, 0]

texts = tf.constant(texts)
labels = tf.constant(labels)



# Vectorizer
vectorizer = TextVectorization(max_tokens=2000, output_sequence_length=20)
vectorizer.adapt(texts)

# Model
model = tf.keras.Sequential([
    vectorizer,
    Embedding(2000, 8),
    GlobalAveragePooling1D(),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(texts, labels, epochs=20, verbose=0)

# Save
model.save("sentiment_model.h5")

print("Saved sentiment_model.h5 successfully!")
