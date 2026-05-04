# Load in all models and store again as JSON, for use with the API
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

#Re-create LSTM Model, Load Model Architecture, and Scaler
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=24, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(24, 2) 
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
lstm_model = LSTMModel()

f = os.path.join(models_dir, "LSTM", "lstm_model.pth")
lstm_model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))
lstm_model.eval()

with open(os.path.join(models_dir, "LSTM", "scaler.pkl"), "rb") as f:
    lstm_scaler = joblib.load(f)  

#Load Mini-Rocket Model
with open(os.path.join(models_dir, "MiniRockets", "minirocket.pkl"), "rb") as f:
    mr_model = pickle.load(f)

with open(os.path.join(models_dir, "MiniRockets", "scaler.pkl"), "rb") as f:
    mr_scaler = pickle.load(f)

with open(os.path.join(models_dir, "MiniRockets", "sgdc.pkl"), "rb") as f:
    mr_sgdc = pickle.load(f)

# Load data
def loadData():
    dataDir = os.path.abspath(os.path.join(base_dir,"..", "data","dataverse_files"))
    # One partition at a time
    partitions_list = [
        'partition5_instances']
    
    all_files = []
    for partition_inst in partitions_list:
        partition_inst_path = os.path.abspath(os.path.join(dataDir,partition_inst))
        partition_num = partition_inst.replace('_instances', '')
        partition_folder_path = os.path.abspath(os.path.join(partition_inst_path,partition_num))
        for folder in ['FL', 'NF']:
            folder_path = os.path.abspath(os.path.join(partition_folder_path,folder))
            for file in os.listdir(folder_path):
                fullPath = os.path.abspath(os.path.join(folder_path,file))
                all_files.append((fullPath, file, folder, partition_num))
    return all_files

allFiles = loadData()

# Convert models to JSON
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

def JSONLSTM():
    modelJSON = json.dumps({"LSTM":{"input_size":6, "hidden_size":24, "batch_first":True},
                 "dropout":{"p":0.5, "inplace":False},
                 "linear":{"in_features":24, "out_featurs":2,"bias":True}})
    scalerJSON = json.dumps({
        "mean":list(lstm_scaler.mean_),
        "var":list(lstm_scaler.var_),
        "n_features_in":int(lstm_scaler.n_features_in_),
        "n_samples_seen":int(lstm_scaler.n_samples_seen_)
    })
    with open("LSTMModel.json","w") as f:
        f.write(modelJSON)
    with open("LSTMScaler.json","w") as f:
        f.write(scalerJSON)
# Preprocessing and prediction functions from original notebooks

def preprocessing_XGBoost(df):
    def lwma_linear_fast(df, column):
            x = np.asarray(df[column].to_numpy(), dtype=np.float64)
            n = x.size
            if n == 0:
                return np.nan

            i = np.arange(1, n + 1, dtype=np.float64)          # still allocates, but small + fast
            denom = n * (n + 1) / 2.0                          # sum(i)
            return (x @ i) / denom

    def lwma_quadratic_fast(df, column):
        x = np.asarray(df[column].to_numpy(), dtype=np.float64)
        n = x.size
        if n == 0:
            return np.nan

        i = np.arange(1, n + 1, dtype=np.float64)
        denom = n * (n + 1) * (2 * n + 1) / 6.0            # sum(i^2)
        return (x @ (i * i)) / denom
    
    def average_absolute_change(df,column):
        x = df[column].to_numpy()
        return np.mean(np.abs(np.diff(x)))
    
    def last_value(df,column):
        return df[column].iloc[-1]

    def average_absolute_derivative_change(df, column, dt_minutes = 12.0):
        s = pd.to_numeric(df[column], errors="coerce")
        deriv = s.diff() / dt_minutes
        deriv_change = deriv.diff().abs()
        return float(deriv_change.mean())
    
    feature_list_summaries = [ 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
       'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45',
       'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY',
       'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE']
    
    descriptors = ['Max', 'Min', 'Median', 'Mean', 'Std', 'Variance', 'Skewness', 'Kurtosis']

    # force numeric
    df[feature_list_summaries] = df[feature_list_summaries].apply(pd.to_numeric, errors="coerce")

    # treat inf as missing
    df[feature_list_summaries] = df[feature_list_summaries].replace([np.inf, -np.inf], np.nan)

    # interpolate + fill
    df[feature_list_summaries] = (
        df[feature_list_summaries]
        .interpolate(method="linear", axis=0, limit_direction="both")
        .ffill()
        .bfill()
    )

    # if any columns were entirely NaN, set them to 0
    all_nan_cols = df[feature_list_summaries].columns[df[feature_list_summaries].isna().all()]
    if len(all_nan_cols) > 0:
        df[all_nan_cols] = 0.0

    # final hard assert
    if df[feature_list_summaries].isna().any().any():
        bad = df[feature_list_summaries].columns[df[feature_list_summaries].isna().any()].tolist()
        raise ValueError(f"Still have NaNs after fill in columns: {bad}")

    desc_feats = {}
    for feat in feature_list_summaries:
        s = df[feat]
        n = s.notna().sum()

        desc_feats[f"{feat}_Max"] = float(s.max())
        desc_feats[f"{feat}_Min"] = float(s.min())
        desc_feats[f"{feat}_Median"] = float(s.median())
        desc_feats[f"{feat}_Mean"] = float(s.mean())
        desc_feats[f"{feat}_Std_dev"] = float(s.std(ddof=1)) if n > 1 else 0.0
        desc_feats[f"{feat}_Variance"] = float(s.var(ddof=1)) if n > 1 else 0.0
        desc_feats[f"{feat}_Skewness"] = float(s.skew()) if n > 2 else 0.0
        desc_feats[f"{feat}_Kurtosis"] = float(s.kurt()) if n > 3 else 0.0

        # if these helpers can output nan, wrap them too:
        desc_feats[f"{feat}_avg_abs_derivative_change"] = float(np.nan_to_num(average_absolute_derivative_change(df, feat), nan=0.0))
        desc_feats[f"{feat}_last_value"] = float(np.nan_to_num(last_value(df, feat), nan=0.0))
        desc_feats[f"{feat}_average_absolute_change"] = float(np.nan_to_num(average_absolute_change(df, feat), nan=0.0))
        desc_feats[f"{feat}_quadratic_weighted_moving_average"] = float(np.nan_to_num(lwma_quadratic_fast(df, feat), nan=0.0))
        desc_feats[f"{feat}_linear_weighted_moving_average"] = float(np.nan_to_num(lwma_linear_fast(df, feat), nan=0.0))

    return pd.DataFrame([desc_feats])

def preprocessing_LSTM(df):
    feature_list_summaries = [ 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'R_VALUE']
    df = df[feature_list_summaries].copy()
    # force numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # treat inf as missing
    df = df.replace([np.inf, -np.inf], np.nan)

    # interpolate + fill
    df = (
        df
        .interpolate(method="linear", axis=0, limit_direction="both")
        .ffill()
        .bfill()
    )

    # if any columns were entirely NaN, set them to 0
    all_nan_cols = df.columns[df.isna().all()]
    if len(all_nan_cols) > 0:
        df[all_nan_cols] = 0.0

    # final hard assert
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Still have NaNs after fill in columns: {bad}")
    
    return df.values.astype('float32')

def preprocessing_MiniRocket(df):
    containsWords = ["FLARE", "LOC", "LABEL", "Timestamp"]

    dropColumns = []

    for col in df.columns:
        for word in containsWords:
            if word in col:
                dropColumns.append(col)
                break

    sample =  df.drop(columns=dropColumns)
    return sample.values

# Prediction methods from flask app
def predMiniRockets(df):
        X = preprocessing_MiniRocket(df) #Prepare data for MiniRockets model
        print("preprocessed")
        X = X.reshape(-1,60,38) #Reshape data into shape expected by MiniRockets

        X_mr = mr_model.transform(X) #Transform data through MiniRockets
        X_scaled = mr_scaler.transform(X_mr) #MiniRockets Scaler

        probs = mr_sgdc.predict_proba(X_scaled) #Predict probabilities 

        return probs

def predXGBoost(df):
        
        df_scal = xgb_scaler.transform(df) #XGBoost scaler
        probs = xgb_model.predict_proba(df_scal) #Predict Probabilities of each class
        return probs

def predLSTM(df):
        df_pre = preprocessing_LSTM(df) #Prepare data for LSTM model
        print("preprocessed")
        df_scaled = lstm_scaler.transform(df_pre) #LSTM Scaler
        df_tensor = torch.tensor(df_scaled, dtype=torch.float32) #Convert the scaled data to a tensor
        df_tensor = df_tensor.reshape(-1,60,6)
        lstm_model.eval() #Set LSTM to evaluation mode

        with torch.no_grad():
            output = lstm_model(df_tensor) #Run prediction
    
        probs = torch.softmax(output, dim=1) #Converts the logits outputted by the LSTM into probabilities
        return probs

# Run for full dataset, convert to JSON, and save
def fullPredMiniRockets(df, allFiles):
    res = predMiniRockets(df)
    resDict = {}
    for i in range(0,len(allFiles)):
        resDict[allFiles[i][1]] = {"NF":res[i][0], "FL":res[i][1]}
    with open("MiniRocketsPredPart5.json", "w") as f:
        json.dump(resDict, f, ensure_ascii=False, indent=4)
        
def fullPredXGBoost(df, allFiles):
    resDict = {}
    preList = []
    for i in range(0,len(allFiles)):
        dfSet = df[60*(i):60*(i+1)]
        dfSet = dfSet.copy(deep=True)
        preSlice = preprocessing_XGBoost(dfSet)
        preSlice = preSlice[xgb_featnames]
        preList.append(preSlice)
    dfPreFull = pd.concat(preList, axis=0, ignore_index=True)
    print("preprocessed")
    res = predXGBoost(dfPreFull)
    for i in range(0,len(allFiles)):
        resDict[allFiles[i][1]] = {"NF":float(res[i][0]), "FL":float(res[i][1])}
    
    with open("XGBoostPredPart5.json", "w") as f:
        json.dump(resDict, f, ensure_ascii=False, indent=4)

def fullPredLSTM(df, allFiles):
    res = predLSTM(df)
    resDict = {}
    for i in range(0,len(allFiles)):
        resDict[allFiles[i][1]] = {"NF":res[i][0].item(), "FL":res[i][1].item()}
    with open("LSTMPredPart5.json", "w") as f:
        json.dump(resDict, f, ensure_ascii=False, indent=4)

def loadCSV(allFiles):
    list = []
    for file in allFiles:
        df = pd.read_csv(file[0], sep="\t", header=0)
        list.append(df)
    fullFrame = pd.concat(list,axis=0, ignore_index=True)
    return fullFrame
    
JSONLSTM()
#JSONXGBoost()
#JSONMiniRockets()

#fullFrame = loadCSV(allFiles)
#print("LSTM:")
#fullPredLSTM(fullFrame, allFiles)
#print("MiniRockets:")
#fullPredMiniRockets(fullFrame, allFiles)
#print("XGBoost:")
#fullPredXGBoost(fullFrame, allFiles)

#print("done")