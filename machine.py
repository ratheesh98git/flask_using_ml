import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("D:/ratheesh_code/data.csv")  


X = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = df['Company'].values  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

joblib.dump(model, "ajith_model.pkl")
joblib.dump(scaler, "ajith_scaler.pkl")

def make_prediction(amount):
    X_input = [[amount, amount, amount, amount, amount]]  

    scaler = joblib.load("ajith_scaler.pkl")

    X_input_scaled = scaler.transform(X_input)

    model = joblib.load("ajith_model.pkl")

    prediction = model.predict(X_input_scaled)

    return prediction

amount = float(input("Enter the amount: "))

prediction = make_prediction(amount)

print("Predicted Company:", prediction)
