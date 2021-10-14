from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('decision_tree_loanprediction5.pkl', 'rb'), encoding='utf-8')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html', prediction_text="",alert_label="alert-primary")


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        home_ownership = request.form['home_ownership']
        loan_amount = request.form['loan_amount']
        interest_rate = request.form['interest_rate']
        grade = request.form['grade']
        annual_income = request.form['annual_income']
        emp_length = request.form['emp_length']

        prediction = model.predict([[
                age,
                home_ownership,
                loan_amount,
                interest_rate,
                grade,
                annual_income,
                emp_length
        ]])
        msg=  "";
        alert_label = "alert-primary"
        output = round(prediction[0], 2)
        if output == 1:
            msg = name + " your loan application was APPROVED!"
            alert_label = "alert-success"
        elif output == 0:
            msg = name + " your loan application was DECLINED!"
            alert_label = "alert-danger"
        return render_template('index.html',
                prediction_text=msg,
                alert_label=alert_label
                )
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


