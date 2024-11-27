from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('../final_model.pkl')

@app.route('/')
def index():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = int(request.form['Pregnancies'])
        glucose_level = float(request.form['Glucose Level'])
        blood_pressure = float(request.form['Blood Pressure'])
        skin_thickness = float(request.form['Skin Thickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['Diabetes Pedigree Function'])
        age = int(request.form['Age'])

        input_data = np.array([[pregnancies, glucose_level, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        prediction_text = 'You are at risk of diabetes!\nPlease consult the doctor!' if prediction == 1 else 'You are not at risk of diabetes.'

        return render_template('index.html', prediction_text=prediction_text)

# 运行 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
