from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import random # here since no sensors available, we will use this to generate random numbers in a range corresponding to the data we have 

app = Flask(__name__)

cors = CORS(app) # allowed all routes for all domains

# Load the saved model
model = joblib.load('./prediction_data_and_model/xgb_model.pkl')
print(model)

@app.route('/predict', methods=['POST'])
def predict():

    i = 1 # an iterator which will give 5 datapoints for each of the 5 parameters empty arrays below
    Hydraulic_Pressure = []
    Hydraulic_Oil_Temperature = []
    Saw_Blade_RPM = []
    Fuel_Consumption = []
    Blade_Sharpness_Level = []

    while i <= 5:
    # the 5 parameter below have ranges of integers used in the dataset provided, giving the model to predict all 3 
    # category of output possible "All good", "Maintenance Due!", "Repair / Replace"
        Hydraulic_Pressure.append(random.randint(145,300))
        Hydraulic_Oil_Temperature.append(random.randint(20,85))
        Saw_Blade_RPM.append(random.randint(700,2700))
        Fuel_Consumption.append(random.randint(10,24))
        Blade_Sharpness_Level.append(random.randint(29,100))
        i += 1
    print(Hydraulic_Pressure)
    # the sensor_array only takes last datapoint of each parameter to make an array fed to the prediction model and relevant color translation 
    sensor_array = [Hydraulic_Pressure[4], Hydraulic_Oil_Temperature[4], Saw_Blade_RPM[4], Fuel_Consumption[4], Blade_Sharpness_Level[4]]
    sensor_data = np.array([sensor_array]) # convert to np array for inputting this to xgboost model

    print(f"{sensor_data.shape} {sensor_data}") # 1 row, 5 columns shape
    
    prediction = int(model.predict(sensor_data))
    
    # here prediction is an int value predicted by the model as 0, 1 or 2. So we translate it to a readable message 
    if prediction == 0:
        prediction = "All good"
    elif prediction == 1:
        prediction = "Maintenance Due!"
    else:
        prediction = "Repair / Replace"

    # here the color alerts logic is stated as strings of "green", "red", "yellow"  
    Hydraulic_Pressure_color="" or "yellow" # defaults to yellow in case of no sensor reading
    if sensor_array[0] in range(180,280):
        Hydraulic_Pressure_color = "green"
    elif sensor_array[0] in range(175,180) or sensor_array[0] in range(280,290):
        Hydraulic_Pressure_color = "yellow"
    elif sensor_array[0] <= 175 or sensor_array[0] >= 290:
        Hydraulic_Pressure_color = "red"

    Hydraulic_Oil_Temperature_color="" or "yellow"
    if sensor_array[1] in range(30,65):
        Hydraulic_Oil_Temperature_color = "green"
    elif sensor_array[1] in range(25,30) or sensor_array[1] in range(65,70):
        Hydraulic_Oil_Temperature_color = "yellow"
    elif sensor_array[1] <= 25 or sensor_array[1] >= 70:
        Hydraulic_Oil_Temperature_color = "red"

    Saw_Blade_RPM_color="" or "yellow"
    if sensor_array[2] in range(800,2500):
        Saw_Blade_RPM_color = "green"
    elif sensor_array[2] in range(750,800) or sensor_array[2] in range(2500,2550):
        Saw_Blade_RPM_color = "yellow"
    elif sensor_array[2] <= 750 or sensor_array[2] >= 2550:
        Saw_Blade_RPM_color = "red"

    Fuel_Consumption_color="" or "yellow"
    if sensor_array[3] in range(10,20):
        Fuel_Consumption_color = "green"
    elif sensor_array[3] in range(20,23):
        Fuel_Consumption_color = "yellow"
    elif sensor_array[3] >= 23:
        Fuel_Consumption_color = "red"

    Blade_Sharpness_Level_color="" or "yellow"
    if sensor_array[4] in range(75,100):
        Blade_Sharpness_Level_color = "green"
    elif sensor_array[4] in range(50,75):
        Blade_Sharpness_Level_color = "yellow"
    elif sensor_array[4] >= 49:
        Blade_Sharpness_Level_color = "red"


    return jsonify({'prediction': prediction, 
                    
                    'color':
                        [{'Hydraulic_Pressure': Hydraulic_Pressure_color, 
                        'Hydraulic_Oil_Temperature': Hydraulic_Oil_Temperature_color,
                        'Saw_Blade_RPM': Saw_Blade_RPM_color,
                        'Fuel_Consumption': Fuel_Consumption_color,
                        'Blade_Sharpness_Level': Blade_Sharpness_Level_color}],

                    'sensor_data':
                        [{'Hydraulic_Pressure': Hydraulic_Pressure, 
                        'Hydraulic_Oil_Temperature': Hydraulic_Oil_Temperature,
                        'Saw_Blade_RPM': Saw_Blade_RPM,
                        'Fuel_Consumption': Fuel_Consumption,
                        'Blade_Sharpness_Level': Blade_Sharpness_Level}],
                    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5002)
