from flask import Flask, render_template, request, flash
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load the trained ML model
model_path = os.path.join(os.path.dirname(__file__), 'model/loan.pkl')
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define the API endpoint for loan prediction
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict_loan():
    try:
        # Get user inputs from the request form
        Dependents = int(request.form["Dependents"])
        Education = int(request.form["Education"])
        Self_Employed = int(request.form["Self_Employed"])
        Income_Annum = float(request.form["Income_Annum"])
        Loan_Amount = float(request.form["Loan_Amount"])
        Loan_Term = float(request.form["Loan_Term"])
        Cibil_Score = float(request.form["Cibil_Score"])

        # Create a dataframe with user inputs
        data = pd.DataFrame(
            {
                "Dependents": [Dependents],
                "Education": [Education],
                "Self_Employed": [Self_Employed],
                "Income_Annum": [Income_Annum],
                "Loan_Amount": [Loan_Amount],
                "Loan_Term": [Loan_Term],
                "Cibil_Score": [Cibil_Score],
            }
        )

        # Use the trained ML model to make predictions
        predictions = model.predict(data)
       
        return render_template('home.html', prediction=True, prediction_result=predictions[0])
       
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return render_template("home.html")


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
