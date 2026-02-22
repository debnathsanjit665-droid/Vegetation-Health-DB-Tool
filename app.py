from flask import Flask, render_template, request
import numpy as np
import sqlite3
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/vegetation_model.h5")

# Create database table
def init_db():
    conn = sqlite3.connect("database/vegetation.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vegetation_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ndvi_mean REAL,
            ndvi_max REAL,
            ndvi_min REAL,
            ndvi_std REAL,
            predicted_class TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        ndvi_mean = float(request.form["ndvi_mean"])
        ndvi_max = float(request.form["ndvi_max"])
        ndvi_min = float(request.form["ndvi_min"])
        ndvi_std = float(request.form["ndvi_std"])

        input_data = np.array([[ndvi_mean, ndvi_max, ndvi_min, ndvi_std]])
        prediction_probs = model.predict(input_data)
        predicted_class_index = np.argmax(prediction_probs)
        confidence = float(np.max(prediction_probs))

        classes = ["Healthy", "Moderate", "Stressed"]
        prediction = classes[predicted_class_index]

        # Save to database
        conn = sqlite3.connect("database/vegetation.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vegetation_records
            (ndvi_mean, ndvi_max, ndvi_min, ndvi_std, predicted_class, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ndvi_mean, ndvi_max, ndvi_min, ndvi_std, prediction, confidence, datetime.now()))
        conn.commit()
        conn.close()

    return render_template("index.html", prediction=prediction, confidence=confidence)

@app.route("/history")
def history():
    conn = sqlite3.connect("database/vegetation.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vegetation_records ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template("history.html", records=records)

if __name__ == "__main__":
    import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))