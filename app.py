from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model from the pickle file
with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_quote', methods=['POST'])
def get_quote():
    age = int(request.form['age'])
    sex = request.form['sex']
    BMI = float(request.form['BMI'])
    smoker = request.form['smoker']
    children = int(request.form['children'])
    region = request.form['region']
    sex_encoding = {'male': 0, 'female': 1}
    smoker_encoding = {'yes': 1, 'no': 0}
    region_encoding={'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
    data = np.array([age, sex_encoding.get(sex), BMI, children, smoker_encoding.get(smoker), region_encoding.get(region)])
    predicted_charge = loaded_model.predict(data.reshape(1, -1))[0]
    predicted_charge_str = f"{predicted_charge:.2f}"
    print(predicted_charge_str)
    return jsonify({'prediction': predicted_charge_str})

if __name__ == '__main__':
    app.run(debug=True)