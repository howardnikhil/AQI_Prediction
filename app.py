# Importing essential libraries
from flask import Flask, request, render_template
import pickle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the Random Forest Classifier model
filename = 'random_forest_regression_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    logging.debug("Serving index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            logging.debug(f"Received form data: {request.form}")
            t = int(request.form['T'])
            tm = int(request.form['TM'])
            tmm = int(request.form['Tm'])
            slp = int(request.form['SLP'])
            h = int(request.form['H'])
            vv = float(request.form['VV'])
            v = float(request.form['V'])
            vm = int(request.form['VM'])
            
            data = np.array([[t, tm, tmm, slp, h, vv, v, vm]])
            my_prediction = classifier.predict(data)
            prediction = my_prediction[0]
            logging.debug(f"Prediction: {prediction}")
            
            return render_template('result.html', prediction=prediction, error=None)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return render_template('result.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)