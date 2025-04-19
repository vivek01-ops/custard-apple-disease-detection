import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# ==== PATH SETUP ====
dataset_path = "dataset"
checkpoint_dir = "model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
model_path = os.path.join(checkpoint_dir, "custard_model.keras")
weights_path = os.path.join(checkpoint_dir, "best_model.weights.h5")  # ✅ Correct extension

# ==== LOAD DATA ====
@st.cache_data
def load_data():
    categories = os.listdir(dataset_path)
    class_names = sorted(categories)
    data = []

    for label, category in enumerate(class_names):
        cat_path = os.path.join(dataset_path, category)
        for img_name in os.listdir(cat_path):
            try:
                img_path = os.path.join(cat_path, img_name)
                img = load_img(img_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                data.append((img_array, label))
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    
    np.random.shuffle(data)
    X, y = zip(*data)
    return np.array(X), np.array(y), class_names

X, y, class_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== MODEL BUILD ====
def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(len(class_names))

# ==== TRAINING ====
def train_model(epochs):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,  # ✅ Correct extension for weights
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
        batch_size=32
    )

    model.save(model_path)  # ✅ Full model saved with .keras extension

    # Plot accuracy and loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label="Train Acc")
    ax[0].plot(history.history['val_accuracy'], label="Val Acc")
    ax[0].legend(); ax[0].set_title("Accuracy")

    ax[1].plot(history.history['loss'], label="Train Loss")
    ax[1].plot(history.history['val_loss'], label="Val Loss")
    ax[1].legend(); ax[1].set_title("Loss")

    st.pyplot(fig)

# ==== LOAD TRAINED MODEL ====
def load_trained_model():
    model = build_model(len(class_names))
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    return model

# ==== PREDICTION ====
def predict_image(image):
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    loaded_model = load_trained_model()
    pred = loaded_model.predict(img_array)
    class_id = np.argmax(pred)
    confidence = np.max(pred) * 100
    return class_names[class_id], confidence

# ==== STREAMLIT APP ====
st.title("🍏 Custard Apple Disease Classifier")

tab1, tab2 = st.tabs(["🔁 Train Model", "🔍 Predict"])

with tab1:
    st.markdown("### 🧠 Train the CNN model on your dataset")
    epochs = st.number_input("Number of Epochs", min_value=0, max_value=None, value="min", step=1, placeholder="No. of Epochs", disabled=False, label_visibility="visible")



    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            train_model(epochs)
        st.success("✅ Training complete and model saved!")
with tab2:
    st.markdown("### 📷 Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=250)

        if st.button("🔍 Predict"):
            disease, confidence = predict_image(img)
            st.success(f"🦠 Predicted Disease: **{disease}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")
