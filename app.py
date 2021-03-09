import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__,
            template_folder="C:\\Users\PK_Rocker$\\all_codes_python\\all_python_codes\Projects\Students_Marks_Prediction\\template")


model = joblib.load("Students_marks_predictor.pkl")

data = pd.DataFrame()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global data

    input_features = [int(x) for x in request.form.values()]
    features_values = np.array(input_features)

    # validating input hours
    if input_features[0] < 1 or input_features[0] > 12:
        return render_template('index.html', predction_text='Please enter a valid value')

    output = model.predict([features_values])[0][0].round(2)

    # input and predicted value store in data then save in csv file
    data = pd.concat([data, pd.DataFrame({'Study Hours': input_features, 'Student Marks': output})])
    print(data)
    data.to_csv('sample_data_from_app.csv')

    return render_template('index.html',
                           prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(
                               output, int(features_values[0])))


if __name__ == '__main__':
    app.run(debug=True)
