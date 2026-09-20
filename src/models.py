"""CNN definition + Grad-CAM."""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(num_classes: int, input_shape=(64, 64, 1)) -> keras.Model:
    inp = keras.Input(shape=input_shape)
    x = inp
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]


def last_conv_name(model: keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("Model has no Conv2D layer")


def gradcam_heatmap(model: keras.Model, x: np.ndarray, class_idx: int | None = None) -> np.ndarray:
    """Grad-CAM heatmap in [0, 1] for one image x of shape (H, W, 1)."""
    grad_model = keras.Model(model.inputs, [model.get_layer(last_conv_name(model)).output, model.output])
    inp = tf.convert_to_tensor(x[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def overlay_heatmap(x: np.ndarray, heat: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a heatmap over the (H, W, 1) image -> RGB uint8."""
    h, w = x.shape[:2]
    heat_u8 = np.uint8(255 * cv2.resize(heat, (w, h)))
    color = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor((x.squeeze() * 255).astype("uint8"), cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 1 - alpha, color, alpha, 0)
