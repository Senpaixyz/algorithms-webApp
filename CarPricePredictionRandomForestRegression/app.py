from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('car_datasets_rfr_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html', prediction_text="")


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        make = request.form['make']
        fuel_type = request.form['fuel_type']
        aspiration = request.form['aspiration']
        num_of_doors = request.form['num_of_doors']
        body_style = request.form['body_style']
        drive_wheels = request.form['drive_wheels']
        engine_location = request.form['engine_location']
        wheel_base = request.form['wheel_base']
        length = request.form['length']
        width = request.form['width']
        height = request.form['height']
        curb_weight = request.form['curb_weight']
        engine_type = request.form['engine_type']
        num_of_cylinders= request.form['num_of_cylinders']
        engine_size = request.form['engine_size']
        fuel_system = request.form['fuel_system']
        compression_ratio = request.form['compression_ratio']
        horsepower = request.form['horsepower']
        peak_rpm = request.form['peak_rpm']
        city_mpg = request.form['city_mpg']
        highway_mpg = request.form['highway_mpg']

        prediction = model.predict([[
            make,    fuel_type,    aspiration,    num_of_doors,    body_style,    drive_wheels,    engine_location,
            wheel_base ,   length,    width,    height ,   curb_weight,    engine_type ,   num_of_cylinders,    engine_size,
            fuel_system,    compression_ratio ,  horsepower,    peak_rpm  ,  city_mpg   , highway_mpg
        ]])
        output = round(prediction[0],2)

        return render_template('index.html',
                prediction_text=("The CAR PRICE prediction was  $" + str(output)),
                )
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


