# TODO- Load in all models and store again as JSON, for use with the API
# Also run a full prediction set across the data
# One block per model, load and store all related components
# Will need to re-run when models are updated
# Not technically a notebook, but probably the best place for this file

import torch
import json
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb


# Load models

base_dir = os.path.dirname(os.path.abspath(__file__))

models_dir = os.path.abspath(
    os.path.join(base_dir,"..", "models")
)

#Load XGBoost Model, Scaler, and Feature Names
# Not sure if should be v1 or v2
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(models_dir, "XGBoost", "XGBoostModel.ubj"))

with open(os.path.join(models_dir, "XGBoost", "scaler.pkl"), "rb") as f:
    xgb_scaler = pickle.load(f)

with open(os.path.join(models_dir, "XGBoost", "feature_names.pkl"), "rb") as f:
    xgb_featnames = pickle.load(f)  

# TODO- Load LSTM here. Recreated within flask app, could copy in at end
with open(os.path.join(models_dir, "LSTM", "scaler.pkl"), "rb") as f:
    lstm_scaler = joblib.load(f)  

#Load Mini-Rocket Model
with open(os.path.join(models_dir, "MiniRockets", "minirocket.pkl"), "rb") as f:
    mr_model = pickle.load(f)

with open(os.path.join(models_dir, "MiniRockets", "scaler.pkl"), "rb") as f:
    mr_scaler = pickle.load(f)

with open(os.path.join(models_dir, "MiniRockets", "sgdc.pkl"), "rb") as f:
    mr_sgdc = pickle.load(f)

def JSONXGBoost():
    xgb_model.save_model("XGBoostModel.json")
    scalerJSON = json.dumps({
        "mean":list(xgb_scaler.mean_),
        "var":list(xgb_scaler.var_),
        "n_features_in":xgb_scaler.n_features_in_,
        "n_samples_seen":xgb_scaler.n_samples_seen_
    })
    with open("XGBoostScaler.json", "w") as f:
        f.write(scalerJSON)

def JSONMiniRockets():
    parameters = mr_model.get_params()
    attributes = {"num_kernels":mr_model.num_kernels_}

    modelJSON = json.dumps({"parameters":parameters,
                 "attributes":attributes})
    with open("MiniRocketsModel.json", "w") as f:
        f.write(modelJSON)
    
    scalerJSON = json.dumps({
        "mean":list(mr_scaler.mean_),
        "var":list(mr_scaler.var_),
        "n_features_in":int(mr_scaler.n_features_in_),
        "n_samples_seen":int(mr_scaler.n_samples_seen_)
    })
    with open("MiniRocketsScaler.json", "w") as f:
        f.write(scalerJSON)
    sgdcParameters = mr_sgdc.get_params()
    sgdcParameters['class_weight'] = {0:0.5086520159197093,1:29.395}
    sgdcAttributes = {"coef":mr_sgdc.coef_.tolist(),
                      "intercept":list(mr_sgdc.intercept_),
                      "n_iter":mr_sgdc.n_iter_,
                      "classes":list(mr_sgdc.classes_),
                      "t_":mr_sgdc.t_,
                      "n_features_in":mr_sgdc.n_features_in_}
    sgdcJSON = json.dumps({"parameters":sgdcParameters,
                          "attributes":sgdcAttributes})
    with open("MiniRocketsSgdc.json", "w") as f:
        f.write(sgdcJSON)

#JSONXGBoost()
JSONMiniRockets()