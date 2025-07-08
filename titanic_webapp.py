from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("titanic_model.pkl")

@app.route('/')
def home():
    return render_template("titanic_form.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values from form
    Pclass = int(request.form['Pclass'])
    Sex = int(request.form['Sex'])  # already encoded
    Age = float(request.form['Age'])
    SibSp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = int(request.form['Embarked'])  # already encoded

    data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    prediction = model.predict(data)[0]
    return render_template("titanic_form.html", prediction_text=f"Survival Prediction: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
