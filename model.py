import joblib
import pandas as pd
import numpy as np
import os


curr_path = os.path.dirname(os.path.realpath(__file__))
xgb_final = joblib.load(curr_path + "/model/xgboost_blueberry_final_model.joblib")


def predict_yield(attributes: np.ndarray):
    """ Returns Blueberry Yield value"""
    # print(attributes.shape) # (1,10)


    pred = xgb_final.predict(attributes)
    print("Yield predicted")

    return pred[0]