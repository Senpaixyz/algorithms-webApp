from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('advertising_logistic_regression.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html', prediction_text="")


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        daily_spent = request.form['daily_spent']
        age = request.form['age']
        income = request.form['income']
        gender = request.form['gender']
        usage = request.form['usage']


        prediction = model.predict([[
                daily_spent,
                age,
                income,
                gender,
                usage
        ]])
        msg=  "";
        output = round(prediction[0], 2)
        if output == 1:
            msg = "This person probabily click some advertistment"
        elif output == 0:
            msg = "This person NOT probabily click some advertistment"
        return render_template('index.html',
                prediction_text=msg,
                )
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


