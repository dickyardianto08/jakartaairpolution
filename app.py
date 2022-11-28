import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from flask import Flask
import os
app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        pm10 = float(request.form['pm10'])
        so2 = float(request.form['so2'])
        co = float(request.form['co'])
        o3 = float(request.form['o3'])
        no2 = float(request.form['no2'])

        val = np.array([pm10,  so2, co, o3, no2])

        final_features = [np.array(val)]
        model_path = os.path.join('models', 'jakarta_polution.sav')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', prediction_text=res)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
