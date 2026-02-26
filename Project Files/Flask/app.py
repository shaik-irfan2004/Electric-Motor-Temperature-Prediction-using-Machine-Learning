from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.save")
transform = joblib.load("transform.save")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        ambient_temp = float(request.form.get("ambient_temp"))
        coolant_temp = float(request.form.get("coolant_temp"))
        motor_speed = float(request.form.get("motor_speed"))
        torque = float(request.form.get("torque"))
        current = float(request.form.get("current"))
        voltage = float(request.form.get("voltage"))

        input_data = np.array([[ambient_temp, coolant_temp, motor_speed, torque, current, voltage]])

        input_scaled = transform.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"<h2>Error Occurred: {e}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
