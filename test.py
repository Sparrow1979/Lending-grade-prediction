import requests

app_url = "https://gradepredict-400511.lm.r.appspot.com/predict"


api_call = {
    "loan_amnt": 10000,
    "term": 36,
    "total_pymnt": 500,
    "emp_length": 10,
    "home_ownership": "RENT",
    "annual_inc": 1000000,
    "purpose": "credit_card",
    "dti": 20,
    "risk_score": 700,
    "history_length": 180,
    "addr_state": "MC",
    "recoveries": 0
}

response = requests.post(app_url, json=api_call)

if response.status_code == 200:

    data = response.json()
    print("Predicted Grades:", data.get("predicted_grades"))
else:
    print("Error:", response.status_code, response.text)
