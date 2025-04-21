import joblib
import pandas as pd
import numpy as np
from neural_network import NeuralNetwork


def predict_purchase(input_values):
    try:
        model = joblib.load('trained_nn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')

        input_df = pd.DataFrame([input_values])
        input_df = pd.get_dummies(input_df, columns=['Month', 'VisitorType'], drop_first=True)

        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[feature_names]

        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)

        return bool(prediction[0])

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None


if __name__ == "__main__":
    sample_input = {
        'Administrative': 3,
        'Administrative_Duration': 87.33,
        'Informational': 0,
        'Informational_Duration': 0,
        'ProductRelated': 27,
        'ProductRelated_Duration': 798.33,
        'BounceRates': 0,
        'ExitRates': 0.012644,
        'PageValues': 22.91,
        'SpecialDay': 0.8,
        'Month': 'Feb',
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 3,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': 0
    }

    result = predict_purchase(sample_input)
    print(f"Will make purchase: {result}")
