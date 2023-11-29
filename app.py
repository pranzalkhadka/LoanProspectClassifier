import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('AdaBoost.pickle','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    input_df = pd.DataFrame(data, index=[0])
    new_data = preprocessor.transform(input_df)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)
