from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__, static_url_path='/static')

model = joblib.load("ajith_model.pkl")  
scaler = joblib.load("ajith_scaler.pkl") 

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    return redirect(url_for('search'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        amount = float(request.form['amount'])
        return redirect(url_for('result', amount=amount))
    return render_template('search.html')

def make_prediction(amount):
    X_input = [[amount, amount, amount, amount, amount]] 
    X_input_scaled = scaler.transform(X_input)
    prediction = model.predict(X_input_scaled)
    return prediction

@app.route('/result/<amount>')
def result(amount):
    prediction = make_prediction(float(amount))
    return render_template('result.html', prediction=prediction, amount=amount)

if __name__ == '__main__':
    app.run(debug=True,host="localhost",port=5000)
