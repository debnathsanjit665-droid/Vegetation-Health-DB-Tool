from flask import Flask, render_template, request, redirect, session
import sqlite3
from datetime import datetime
import uuid
import os

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

DATABASE = "database/vegetation.db"

def init_db():
    if not os.path.exists("database"):
        os.makedirs("database")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ndvi_mean REAL,
        ndvi_max REAL,
        ndvi_min REAL,
        ndvi_std REAL,
        predicted_class TEXT,
        confidence REAL,
        timestamp TEXT,
        session_id TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


@app.route("/", methods=["GET", "POST"])
def index():

    # Unique session per device
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

    prediction = None
    confidence = None

    if request.method == "POST":

        ndvi_mean = float(request.form["ndvi_mean"])
        ndvi_max = float(request.form["ndvi_max"])
        ndvi_min = float(request.form["ndvi_min"])
        ndvi_std = float(request.form["ndvi_std"])

        # Dummy Prediction Logic (You can replace with ML model)
        if ndvi_mean > 0.5:
            prediction = "Healthy Vegetation"
            confidence = 0.92
        else:
            prediction = "Unhealthy Vegetation"
            confidence = 0.78

        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions
        (ndvi_mean, ndvi_max, ndvi_min, ndvi_std, predicted_class, confidence, timestamp, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ndvi_mean,
            ndvi_max,
            ndvi_min,
            ndvi_std,
            prediction,
            confidence,
            timestamp,
            session["user_id"]
        ))

        conn.commit()
        conn.close()

    return render_template("index.html", prediction=prediction, confidence=confidence)



@app.route("/history")
def history():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT ndvi_mean, ndvi_max, ndvi_min, ndvi_std,
           predicted_class, confidence, timestamp
    FROM predictions
    WHERE session_id = ?
    ORDER BY id DESC
    """, (session["user_id"],))

    data = cursor.fetchall()
    conn.close()

    return render_template("history.html", data=data)

@app.route("/reset")
def reset():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE session_id = ?",
        (session["user_id"],)
    )

    conn.commit()
    conn.close()

    return redirect("/history")

if __name__ == "__main__":
    app.run(debug=True)

    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)