import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib

app = Flask(__name__)
model = joblib.load('modelo/model.pkl')


@app.route('/')

def display_gui():
    return render_template('index.html')

@app.route('/verificar', methods=['POST'])
def verificar():
    Pregnancies = request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']
    teste = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    
          
    classe = model.predict(teste)[0]
    prob = model.predict_proba(teste)[0]
    probab = round(np.max(prob)*100,2)
    
    
    return render_template('index.html',classe=str(classe), probab=(probab))

if __name__ == "__main__":
  
    app.run()
    