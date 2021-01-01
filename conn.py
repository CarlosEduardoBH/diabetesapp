import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib

app = Flask(__name__)
model = joblib.load('Modelo/model.pkl')


@app.route("/")

def about():
    return render_template('ml.html')

    
@app.route("/verificar/", methods=['POST'])
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
    
    print(":::::: Dados de Teste :::::::")
    print("Pregnancies: {}".format(Pregnancies))
    print("Glucose: {}".format(Glucose))
    print("BloodPressure: {}".format(BloodPressure))
    print("SkinThickness: {}".format(SkinThickness))
    print("Insulin: {}".format(Insulin))
    print("BMI: {}".format(BMI))
    print("DiabetesPedigreeFunction: {}".format(DiabetesPedigreeFunction))
    print("Age: {}".format(Age))
    print("\n")
    
        
    classe = model.predict(teste)[0]
    prob = model.predict_proba(teste)[0]
    probab = round(np.max(prob)*100,2)
    
    print("Classe Predita: {}".format(str(classe)))
    print("A probalidade é de: {}".format(probab))

    return render_template('ml.html',classe=str(classe), probab=(probab))

if __name__ == "__main__":
  
    port = os.environ.get('port',5000)
    app.run(host='0.0.0.0',port=port)
    