from flask import Flask, render_template, request
from xgboost import XGBClassifier
import joblib
import pandas as pd

app = Flask(__name__)

model = XGBClassifier()
model.load_model("best_model.json")
le = joblib.load("label_encoder.pkl")

# The 8 raw numeric inputs collected from the form
numeric_features = ['u', 'g', 'r', 'i', 'z', 'redshift', 'alpha', 'delta']

# EXACT training column list & order, taken from your notebook's X_train.columns output
model_columns = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift',
                  'spectral_type_G/K', 'spectral_type_M', 'spectral_type_O/B',
                  'galaxy_population_Red_Sequence']

# 4 real spectral_type categories (A/F was dropped as baseline by drop_first=True)
spectral_options = ['A/F', 'G/K', 'M', 'O/B']

# TODO: confirm the exact spelling of the dropped baseline category by running
# print(train['galaxy_population'].unique()) in Colab, then update the first value below
galaxy_options = ['Blue_Cloud', 'Red_Sequence']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            input_dict = {feat: float(request.form[feat]) for feat in numeric_features}
            spectral_type = request.form['spectral_type']
            galaxy_population = request.form['galaxy_population']

            # start with a single row of all zeros, using the EXACT training columns/order
            input_df = pd.DataFrame([[0] * len(model_columns)], columns=model_columns)

            # fill in the numeric values
            for feat, val in input_dict.items():
                if feat in input_df.columns:
                    input_df[feat] = val

            # set the matching one-hot column to 1 (stays all-zero if the user picked
            # the baseline/dropped category, which is correct behavior)
            spec_col = f'spectral_type_{spectral_type}'
            if spec_col in input_df.columns:
                input_df[spec_col] = 1

            gal_col = f'galaxy_population_{galaxy_population}'
            if gal_col in input_df.columns:
                input_df[gal_col] = 1

            # enforce column order to match training exactly
            input_df = input_df[model_columns]

            pred_num = model.predict(input_df)[0]
            prediction = le.inverse_transform([pred_num])[0]

        except Exception as e:
            error = str(e)

    return render_template('index.html', numeric_features=numeric_features,
                            spectral_options=spectral_options, galaxy_options=galaxy_options,
                            prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
