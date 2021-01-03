import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib

app = Flask(__name__)
model = joblib.load('Modelo/model.pkl')


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


def import_module(name, package=None):

    """Import a module.
    The 'package' argument is required when performing a relative import. It
    specifies the package to use as the anchor point from which to resolve the
    relative import to an absolute import.

    """
    level = 0
    if name.startswith('.'):
        if not package:

            msg = ("the 'package' argument is required to perform a relative "
                   "import for {!r}")
            raise TypeError(msg.format(name))
        for character in name:
            if character != '.':

                break
            level += 1
    return _bootstrap._gcd_import(name[level:], package, level)


if __name__ == "__main__":
  
    app.run()
    