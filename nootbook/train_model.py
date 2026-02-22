import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

# Generate Synthetic NDVI Dataset
np.random.seed(42)
data_size = 1000

ndvi_mean = np.random.uniform(0, 1, data_size)
ndvi_max = np.random.uniform(0, 1, data_size)
ndvi_min = np.random.uniform(0, 1, data_size)
ndvi_std = np.random.uniform(0, 0.2, data_size)

df = pd.DataFrame({
    "ndvi_mean": ndvi_mean,
    "ndvi_max": ndvi_max,
    "ndvi_min": ndvi_min,
    "ndvi_std": ndvi_std
})

# Label Rule
def label_ndvi(value):
    if value > 0.6:
        return "Healthy"
    elif value >= 0.3:
        return "Moderate"
    else:
        return "Stressed"

df["label"] = df["ndvi_mean"].apply(label_ndvi)

# Preprocessing
X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# ANN Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save Model
os.makedirs("/model", exist_ok=True)
model.save("/model/vegetation_model.h5")

print("Model Saved Successfully!")