from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
app = Flask(__name__)
model = pickle.load(open('diabetes_prediction_knn.pkl', 'rb'), encoding='utf-8')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html', prediction_text="",alert_label="alert-primary")


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        name = request.form['Name']

        prediction = model.predict([[
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age
        ]])
        msg=  "";
        alert_label = "alert-primary"
        output = round(prediction[0], 2)
        if output == 1:
            msg = name + " you are more likely to be Diabetic"
            alert_label = "alert-success"
        elif output == 0:
            msg = name + " you are not likely to be Diabetic"
            alert_label = "alert-danger"
        return render_template('index.html',
                prediction_text=msg,
                alert_label=alert_label
                )
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


