import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import category_encoders as ce

import logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

with open('model_steps.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    test = model['model']

grade_mapping = {
    6: 'A',
    5: 'B',
    4: 'C',
    3: 'D',
    2: 'E',
    1: 'F',
    0: 'G'
}

categorical_transformer = test.named_steps['preprocessor'].transformers_[1][1]


@app.route('/predict', methods=['POST'])
def grade_predict():
    try:
        api_call = request.json

        api_call_df = pd.DataFrame([api_call])

        categorical_features_api = ["home_ownership", "purpose", "addr_state"]

        categorical_transformer = test['preprocessor'].transformers_[1][1]
        api_call_df[categorical_features_api] = categorical_transformer.transform(
            api_call_df[categorical_features_api])

        y_pred = test.predict(api_call_df)

        predicted_letter_grades = [grade_mapping.get(
            grade, 'Unknown') for grade in y_pred]

        logging.info("Predicted Grades: %s", predicted_letter_grades)

        return jsonify({"predicted_grades": predicted_letter_grades})

    except Exception as e:
        logging.error("Error: %s", str(e))
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
