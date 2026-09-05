from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('xgb_stellar_model.pkl')
label_enc = joblib.load('label_encoder.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Stellar Classifier</title></head>
<body style="font-family: Arial; margin: 40px;">
    <h2>Stellar Object Classification Web App</h2>
    <form action="/predict" method="post">
        <label>Redshift:</label><br>
        <input type="text" name="redshift" value="0.42"><br><br>
        <label>u magnitude:</label><br>
        <input type="text" name="u" value="23.66"><br><br>
        <label>g magnitude:</label><br>
        <input type="text" name="g" value="21.95"><br><br>
        <label>r magnitude:</label><br>
        <input type="text" name="r" value="21.08"><br><br>
        <label>i magnitude:</label><br>
        <input type="text" name="i" value="20.18"><br><br>
        <label>z magnitude:</label><br>
        <input type="text" name="z" value="19.20"><br><br>
        <input type="submit" value="Classify Stellar Object">
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.form
    redshift = float(data['redshift'])
    u, g, r, i, z = float(data['u']), float(data['g']), float(data['r']), float(data['i']), float(data['z'])
    
    
    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z
    u_r = u - r
    ug_ratio = u / (g + 1e-6)
    gr_ratio = g / (r + 1e-6)
    
    input_features = np.array([[0, 0, u, g, r, i, z, redshift, u_g, g_r, r_i, i_z, u_r, ug_ratio, gr_ratio] + [0]*10])
    
    prediction = model.predict(input_features)
    predicted_class = label_enc.inverse_transform(prediction)[0]
    
    return f"<h1>Predicted Stellar Class: <span style='color:blue;'>{predicted_class}</span></h1><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)