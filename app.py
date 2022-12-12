from flask import Flask, render_template, Response
from flask_restful import reqparse, Api
import flask

import numpy as np
import pandas as pd
import joblib
import ast

import os
import json

from model import  predict_yield

curr_path = os.path.dirname(os.path.realpath(__file__))


feat_cols = ['clonesize', 'honeybee', 'osmia', 'MinOfUpperTRange', 
       'MaxOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset',
       'fruitmass', 'seeds']

context_dict = {
    'feats': feat_cols,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list,
}


# # INITIATE THE APPLICATION
app = Flask(__name__)
api = Api(app)


# # FOR FORM PARSING
parser = reqparse.RequestParser()
parser.add_argument('list', type=list)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = flask.request.form.get('single_input')

    # "[1,2,3,4]" => [1, 2, 3, 4]
    # "0" => 0

    i = ast.literal_eval(data)
    y_pred = predict_yield(np.array(i).reshape(1,-1))
    # [0]
    
    # json: javascript object notation
    # { "" : "" }
    
    return {"message": "Success", "pred": json.dumps(int(y_pred))}


@app.route('/')
def index():

    # For a singular response object
    # return Response("<h1> Send post request to /api/predict </h1>")


    # For serving front-end files
    return render_template('index.html', **context_dict)

@app.route('/predict', methods=["POST"])
def predict():

#   # for file datatype
    # csv_file = flask.request.files['csv_file']
    # test_data = pd.read_csv(csv_file)

    # flask.request.form.keys() will print all the input from form
    test_data = []
    for val in flask.request.form.values():
        test_data.append(float(val))
    test_data = np.array(test_data).reshape(1,-1)

    y_pred = predict_yield(test_data)
    context_dict['pred']= y_pred

    print(y_pred)

    return render_template('index.html', **context_dict)

if __name__ == "__main__":
    app.run()