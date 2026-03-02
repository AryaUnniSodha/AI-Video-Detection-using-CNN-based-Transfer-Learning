import tensorflow as tf
import numpy as np
import cv2

model = None

def load_model():
    global model

    if model is None:
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    return model


def predict_frame(frame):
    model = load_model()

    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32")
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)

    prediction = model.predict(frame, verbose=0)[0][0]

    return float(prediction)