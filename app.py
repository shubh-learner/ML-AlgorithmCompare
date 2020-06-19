# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Naive Bayes CLassifier model
filename = 'NB-model.pkl'
classifierNB = pickle.load(open(filename, 'rb'))

# Load the Random Forest CLassifier model
filename = 'RFC-model.pkl'
classifierRFC = pickle.load(open(filename, 'rb'))

# Load the KNN CLassifier model
filename = 'KNN-model.pkl'
classifierKNN = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/comparealgo', methods=['POST'])
def comparealgo():
    flag = 'ca'
    return render_template('result.html', flag= flag)
    
@app.route('/naivebayes', methods=['POST'])
def naivebayes():
     flag = 'nb'
     return render_template('index.html', flag= flag)
    
@app.route('/randomforest', methods=['POST'])
def randomforest():
    flag = 'rfc'
    return render_template('index.html',flag= flag)
    
@app.route('/knnclassification', methods=['POST'])
def knnclassification():
    flag = 'knn'
    return render_template('index.html', flag= flag)
 
    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        
        if request.form['predictflag'] == 'nb':
            my_prediction = classifierNB.predict(data)
            return render_template('result.html', prediction=my_prediction, flagpred= 'nb')
        
        elif request.form['predictflag'] == 'rfc':
            my_prediction = classifierRFC.predict(data)
            return render_template('result.html', prediction=my_prediction, flagpred = 'rfc')
        
        elif request.form['predictflag'] == 'knn':
            my_prediction = classifierKNN.predict(data)
            return render_template('result.html', prediction=my_prediction, flagpred = 'knn')
        

if __name__ == '__main__':
	#app.run(debug=True)
    app.run(host='0.0.0.0' , port = 5000)
    