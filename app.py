# Importing essential libraries
from flask import Flask, render_template, request, session
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes-type-predictor1.pkl'
rfc = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
global_prediction = None
global_user_input = None


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
    global global_prediction,global_user_input
    if request.method == 'POST':
        # Extract input values from the form with default values
        name = request.form.get('name')
        age = int(request.form.get('age')) 
        sex = int(request.form.get('sex')) if request.form.get('sex') else 1
        pregnancies = int(request.form.get('pregnancies'))  if request.form.get('pregnancies') else 0
        glucose = float(request.form.get('glucose')) if request.form.get('glucose') else 120.89453125
        blood_pressure = float(request.form.get('bloodpressure')) if request.form.get('bloodpressure') else 115.98145285935
        skin_thickness = float(request.form.get('skinthickness')) if request.form.get('skinthickness') else 20.5364583333333
        bmi = float(request.form.get('bmi')) if request.form.get('bmi') else 28.1492545889596
        dpf = float(request.form.get('dpf')) if request.form.get('dpf') else 0.471876302083333
        weight = float(request.form.get('weight')) if request.form.get('weight') else 38.8545751633986
        height = float(request.form.get('height')) if request.form.get('height') else 1.34934640522875
        family_history = int(request.form.get('familyHistory')) if request.form.get('familyHistory') else 0 
        hba1c = float(request.form.get('hba1c')) if request.form.get('hba1c') else 0
        autoantibodies = int(request.form.get('autoantibodies')) if request.form.get('autoantibodies') else 0
        insulin_taken = int(request.form.get('insulinTaken')) if request.form.get('insulinTaken') else 0
        hypoglycemia = int(request.form.get('hypoglycemia')) if request.form.get('hypoglycemia') else 0
        gestation_previous_pregnancy = int(request.form.get('gestation')) if request.form.get('gestation') else 0
        hdl = float(request.form.get('hdl')) if request.form.get('hdl') else 46.4718700475435
        pcos = int(request.form.get('pcos')) if request.form.get('pcos') else 0
        sedentary_lifestyle = int(request.form.get('sedentaryLifestyle')) if request.form.get('sedentaryLifestyle') else 0
        prediabetes = int(request.form.get('prediabetes')) if request.form.get('prediabetes') else 0

        # Create the input data array
        data = np.array([[age, sex, hba1c, height, weight, bmi, autoantibodies, insulin_taken, hypoglycemia,
                          family_history, pregnancies, glucose, blood_pressure, skin_thickness,
                          dpf, gestation_previous_pregnancy, hdl, pcos, sedentary_lifestyle, prediabetes]])

        my_prediction = rfc.predict(data)  
        global_prediction = my_prediction
        
        user_input = {
            'name':name,
            'age': age,
            'sex': sex,
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'bmi': bmi,
            'dpf': dpf,
            'weight': weight,
            'height': height,
            'family_history': family_history,
            'hba1c': hba1c,
            'autoantibodies': autoantibodies,
            'insulin_taken': insulin_taken,
            'hypoglycemia': hypoglycemia,
            'gestation_previous_pregnancy': gestation_previous_pregnancy,
            'hdl': hdl,
            'pcos': pcos,
            'sedentary_lifestyle': sedentary_lifestyle,
            'prediabetes': prediabetes
        }
        global_user_input = user_input
        
       

        return render_template('result.html', prediction=my_prediction, user_input=user_input)

# Modify your Flask app to handle the /report-card endpoint
@app.route('/report-card')
def report_card():
    global global_prediction, global_user_input  # Access the global variables

    if global_prediction is not None and global_user_input is not None:
        # Pass prediction and user input data to the report card template
        return render_template('report_card.html', prediction=global_prediction, user_input=global_user_input)
    else:
        # If prediction or user input data is not available, handle it accordingly
        return "Prediction or user input data not found."

#if __name__ == '__main__':
#	app.run(debug=True)