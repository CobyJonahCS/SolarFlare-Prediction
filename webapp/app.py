import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #This line was temp fix for some weird conda thing I dont remember

from flask import Flask, render_template, request
from flask_cors import CORS
from flask_restful import Resource, Api
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
import re
import json
from flasgger import Swagger
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#Directories for the models
base_dir = os.path.dirname(os.path.abspath(__file__))

models_dir = os.path.abspath(
    os.path.join(base_dir,"..", "models")
)

#Load XGBoost Model, Scaler, and Feature Names
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

#Load LightGBM Forecasters
lightgbmdir = os.path.join(models_dir, "LightGBMForecasters")
ABSNJZH_model = joblib.load(os.path.join(lightgbmdir, "ABSNJZH_lgbm.pkl"))
R_VALUE_model = joblib.load(os.path.join(lightgbmdir, "R_VALUE_lgbm.pkl"))
TOTBSQ_model = joblib.load(os.path.join(lightgbmdir, "TOTBSQ_lgbm.pkl"))
TOTPOT_model = joblib.load(os.path.join(lightgbmdir, "TOTPOT_lgbm.pkl"))
TOTUSJH_model = joblib.load(os.path.join(lightgbmdir, "TOTUSJH_lgbm.pkl"))
TOTUSJZ_model = joblib.load(os.path.join(lightgbmdir, "TOTUSJZ_lgbm.pkl"))

#Pre-Processing Functions for each Model. Each function can be found in detail in the notebooks.
#XGBoost pre-processing can be found in data_cleaning.ipynb
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
#XGBoost pre-processing can be found in LSTM_notebook.ipynb
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
#XGBoost pre-processing can be found in productionise_mr.ipynb
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

#Forecaster pre-processing can be found in ligthGBMforcaster.ipynb
def build_features_for_file(temp_df, feature_cols, lags, rolling_windows):
    #rolling only use prev
    feature_dict = {}

    for feature in feature_cols:
        s = temp_df[feature]

        # lags: past only
        for lag in lags:
            feature_dict[f"{feature}_lag_{lag}"] = s.shift(lag)

        # rolling only past rows
        shifted = s.shift(1)
        for window in rolling_windows:
            feature_dict[f"{feature}_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
            feature_dict[f"{feature}_roll_std_{window}"] = shifted.rolling(window=window, min_periods=1).std()

    return pd.DataFrame(feature_dict, index=temp_df.index)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    #Grab csv file and requested model from prediction form
    csv = request.files['file']
    model = request.form['model']

    print(csv.filename)
    print(model)

    #Read csv file into a pandas dataframe
    df = pd.read_csv(csv, sep="\t")

    #XGBoost Prediction
    if model == "XGBoost":
        df_pre = preprocessing_XGBoost(df) #Prepare data for XGBoost model
        df_pre = df_pre[xgb_featnames] #Take only the feature names the XGBoost expects. Preprocessing does this already, this is more like a sanity check.

        df_scal = xgb_scaler.transform(df_pre) #XGBoost scaler

        pred = xgb_model.predict(df_scal)[0] #Predict class
        probs = xgb_model.predict_proba(df_scal) #Predict Probabilities of each class
        probs_df = pd.DataFrame(probs, columns=["NF", "F"], index=["Probability"]) #Create a dataframe holding probabilities of each class

        #Return the prediction and a table of probabilities as HTML back to the predictions page.
        result = f"<p>Prediction: {pred}</p><div class='feature-table-wrap'>{probs_df.to_html(border=False, classes='feature-table')}</div>"

        return result
    #LSTM Prediction
    if model == "LSTM":
        df_pre = preprocessing_LSTM(df) #Prepare data for LSTM model
        df_scaled = lstm_scaler.transform(df_pre) #LSTM Scaler
        df_tensor = torch.tensor(df_scaled, dtype=torch.float32) #Convert the scaled data to a tensor
        df_tensor = df_tensor.unsqueeze(0) #Change shape to (1, 60, 5) as the csv is a single sample with 60 timesteps and 5 features.

        lstm_model.eval() #Set LSTM to evaluation mode

        with torch.no_grad():
            output = lstm_model(df_tensor) #Run prediction
    
        probs = torch.softmax(output, dim=1) #Converts the logits outputted by the LSTM into probabilities
        pred = torch.argmax(probs, dim=1).item() #Chooses the class with the highest probability

        probs_df = pd.DataFrame(probs, columns=["NF", "F"], index=["Probability"]) #Create a dataframe holding probabilities of each class

        #Return the prediction and a table of probabilities as HTML back to the predictions page.
        result = f"<p>Prediction: {pred}</p><div class='feature-table-wrap'>{probs_df.to_html(border=False, classes='feature-table')}</div>"

        return result
    #MiniRockets Prediction
    if model == "MiniRocket":
        X = preprocessing_MiniRocket(df) #Prepare data for MiniRockets model
        
        X = X.reshape(1,60,38) #Reshape data into shape expected by MiniRockets

        X_mr = mr_model.transform(X) #Transform data through MiniRockets
        X_scaled = mr_scaler.transform(X_mr) #MiniRockets Scaler

        probs = mr_sgdc.predict_proba(X_scaled) #Predict probabilities 
        pred = int(probs[0,1] >= 0.9) #Only predicts class 1 if the probability is above the threshold of 0.9
        probs_df = pd.DataFrame(probs, columns=["NF", "F"], index=["Probability"]) #Create a dataframe holding probabilities of each class

        #Return the prediction and a table of probabilities as HTML back to the predictions page.
        result = f"<p>Prediction: {pred}</p><div class='feature-table-wrap'>{probs_df.to_html(border=False, classes='feature-table')}</div>"

        return result

    result = f"Model: {model}<br>Rows: {len(df)}"

    return result

@app.route('/forecast', methods=['POST'])
def forecast():
    FEATURE_COLS = [ # these are the features that we need to forecast 
        "TOTUSJH",
        "TOTBSQ",
        "TOTPOT",
        "TOTUSJZ",
        "ABSNJZH",
        "R_VALUE",
    ]

    LAGS            = list(range(1, 25))   # past 24 steps (4.8 hrs) as input
    ROLLING_WINDOWS = [3, 6, 9]

    #Grab csv file and requested model from prediction form
    csv = request.files['file']

    print(csv.filename)

    #Read csv file into a pandas dataframe
    df = pd.read_csv(csv, sep="\t")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    ar_match = re.search(r"_ar(\d+)", csv.filename)
    df["active_region"] = int(ar_match.group(1)) if ar_match else pd.NA
    df["source_file"] = csv.filename

    # interpolate raw features only
    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .interpolate(method="linear", axis=0, limit_direction="both")
        .ffill()
        .bfill()
    )

    features = build_features_for_file(df, feature_cols=FEATURE_COLS, lags=LAGS, rolling_windows=ROLLING_WINDOWS)

    new_df = pd.concat([df, features], axis=1)
    X = new_df.copy()
    predictions = {}

    #Predictions for each feature with their model.
    predictions["ABSNJZH"] = ABSNJZH_model.predict(X[ABSNJZH_model.feature_name_].to_numpy())
    predictions["R_VALUE"] = R_VALUE_model.predict(X[R_VALUE_model.feature_name_].to_numpy())
    predictions["TOTBSQ"] = TOTBSQ_model.predict(X[TOTBSQ_model.feature_name_].to_numpy())
    predictions["TOTPOT"] = TOTPOT_model.predict(X[TOTPOT_model.feature_name_].to_numpy())
    predictions["TOTUSJH"] = TOTUSJH_model.predict(X[TOTUSJH_model.feature_name_].to_numpy())
    predictions["TOTUSJZ"] = TOTUSJZ_model.predict(X[TOTUSJZ_model.feature_name_].to_numpy())

    #Turn into dataframe
    output_df = pd.DataFrame.from_dict(predictions)

    #Use timestamps as index
    times = pd.date_range(
        start=df["Timestamp"].iloc[-1],
        periods=len(output_df)+1,
        freq="12min"
    )[1:]
    output_df.index = times + pd.Timedelta(minutes=12) #Shift timestamps forward by 1 timestep, as the predictions are

    #Plotly subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        vertical_spacing=0.3,
        horizontal_spacing=0.08,
        subplot_titles=FEATURE_COLS
    )

    #Fill the subplots with values
    count = 0
    for r in range(1,3):
        for c in range(1,4):
            fig.add_trace(
                go.Scatter(x=output_df.index, y=output_df[FEATURE_COLS[count]].values.tolist()),
                row=r, col=c
            )
            count += 1

    #Add titles, and remove pointless key/legend
    fig.update_layout(
        title="Forecasting Predictions for file: "+csv.filename,
        showlegend=False
    )

    #Convert figure to html
    graph = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    predicted_values = output_df.tail(1) #Get the last value, the predicted 61st timestep

    #Output the result, with the predicted values turned into a html table
    result = f"<p>Forecast:</p><div class='feature-table-wrap'>{predicted_values.to_html(border=False, classes='feature-table')}</div><div>{graph}</div>"
    return result

#@app.route("/")
#def home():
#    return render_template('index.html')

#@app.route("/xgboost.html")
#def XGBoost():
#    return render_template("xgboost.html")

#@app.route("/LSTM.html")
#def LSTM():
#    return render_template("LSTM.html")

#@app.route("/minirocket.html")
#def miniRocket():
#    return render_template("miniRocket.html")

#@app.route("/Predictions.html")
#def predictionsPage():
#    return render_template("Predictions.html")

#@app.route("/forecasting.html")
#def forecastinPage():
#    return render_template("forecasting.html")

#@app.route("/APIDocs.html")
#def APIDocs():
#    return render_template("APIDocs.html")

#@app.route("/LightGBM.html")
#def LightGBM():
#    return render_template("LightGBM.html")

#@app.route("/feature_descriptions.html")
#def feature_descriptions():
#    return render_template("feature_descriptions.html")

# API page management
app.config['SWAGGER'] = {
    'title': 'Solar Flare Prediction API',
    'uiversion': 3,
    'version':"1.0"
}
swagger = Swagger(app)
api = Api(app)

# Load in the forecaster parameters (change when forecast predictions page changed)
with open(os.path.join(models_dir, "LightGBMForecasters", "ABSNJZH_lgbm.pkl"), "rb") as f:
    ABSNJZH = (pickle.load(f)).booster_.dump_model()
with open(os.path.join(models_dir, "LightGBMForecasters", "R_VALUE_lgbm.pkl"), "rb") as f:
    R_VALUE = (pickle.load(f)).booster_.dump_model()
with open(os.path.join(models_dir, "LightGBMForecasters", "TOTBSQ_lgbm.pkl"), "rb") as f:
    TOTBSQ = (pickle.load(f)).booster_.dump_model()
with open(os.path.join(models_dir, "LightGBMForecasters", "TOTPOT_lgbm.pkl"), "rb") as f:
    TOTPOT = (pickle.load(f)).booster_.dump_model()
with open(os.path.join(models_dir, "LightGBMForecasters", "TOTUSJH_lgbm.pkl"), "rb") as f:
    TOTUSJH = (pickle.load(f)).booster_.dump_model()
with open(os.path.join(models_dir, "LightGBMForecasters", "TOTUSJZ_lgbm.pkl"), "rb") as f:
    TOTUSJZ = (pickle.load(f)).booster_.dump_model()

# Return all forecasters. Should simplify to only give one at a time
class getLightGBM(Resource):
    def get(self, feature):
        """
        Get the LightGBM forecasters
        ---
        tags:
            - Get model
        description: Returns the full set of LightGBM forecasters, one for each of the observed parameters
        parameters: [{
            "name": "feature",
            "description": "The desired model. Will return all on empty input\n
            Valid inputs: ABSNJZH, R_VALUE, TOTBSQ, TOTPOT, TOTUSJH, TOTUSJZ",
            "allowMultiple": False,
            "in": path,
            "type": "string"}]
        responses:
            200:
                description: Format of a successful response. The models will be returned instead of just the names
                examples:
                    application/json: "{ABSNJZH:ABSNJZH,
                        \nR_VALUE:R_VALUE,
                        \nTOTBSQ:TOTBSQ,
                        \nTOTPOT:TOTPOT,
                        \nTOTUSJH:TOTUSJH,
                        \nTOTUSJZ:TOTUSJZ}"

        """
        if feature:
            if str(feature).upper() == "ABSNJZH":
                return {"ABSNJZH":ABSNJZH}
            elif str(feature).upper() == "R_VALUE":
                return {"R_VALUE":R_VALUE}
            elif str(feature).upper() == "TOTBSQ":
                return {"TOTBSQ":TOTBSQ}
            elif str(feature).upper() == "TOTPOT":
                return {"TOTPOT":TOTPOT}
            elif str(feature).upper() == "TOTUSJH":
                return {"TOTUSJH":TOTUSJH}
            elif str(feature).upper() == "TOTUSJZ":
                return {"TOTUSJZ":TOTUSJZ}
            elif str(feature).upper() == "UNDEFINED":
                return {"ABSNJZH":ABSNJZH,
                            "R_VALUE":R_VALUE,
                            "TOTBSQ":TOTBSQ,
                            "TOTPOT":TOTPOT,
                            "TOTUSJH":TOTUSJH,
                            "TOTUSJZ":TOTUSJZ}
            else:
                return []
        else:
            return {"ABSNJZH":ABSNJZH,
                        "R_VALUE":R_VALUE,
                        "TOTBSQ":TOTBSQ,
                        "TOTPOT":TOTPOT,
                        "TOTUSJH":TOTUSJH,
                        "TOTUSJZ":TOTUSJZ}

# Get the JSON versions of all models
with open(os.path.join(models_dir, "XGBoost", "XGBoostModel.json"), "rb") as f:
    XGBoostJson = json.load(f)
with open(os.path.join(models_dir, "XGBoost", "XGBoostScaler.json"), "rb") as f:
    XGBoostJsonScaler = json.load(f)
with open(os.path.join(models_dir, "MiniRockets", "MiniRocketsModel.json"), "rb") as f:
    miniRocketsJson = json.load(f)
with open(os.path.join(models_dir, "MiniRockets", "MiniRocketsScaler.json"), "rb") as f:
    miniRocketsJsonScaler = json.load(f)
with open(os.path.join(models_dir, "MiniRockets", "MiniRocketsSgdc.json"), "rb") as f:
    miniRocketsJsonSgdc = json.load(f)
with open(os.path.join(models_dir, "LSTM", "LSTMModel.json"), "rb") as f:
    LstmJSON = json.load(f) # Ignore formatting change, constarts are defined by all caps
with open(os.path.join(models_dir, "LSTM", "LSTMScaler.json"), "rb") as f:
    LSTMJSONScaler = json.load(f)

class getLSTM(Resource):
    def get(self):
        """
        Get the LSTM model
        ---
        tags:
            - Get model
        description: Returns the LSTM model and all requred parameter files
        responses:
            200:
                description: Format of a successful response. The model will be returned instead of just the name
                examples:
                    application/json: "{Model:LSTM,\nScaler:Scaler}"

        """
        return {"Model":LstmJSON,
                "Scaler":LSTMJSONScaler}

class getMiniRockets(Resource):
    def get(self):
        """
        Get the MiniRockets model
        ---
        tags:
            - Get model
        description: Returns the Minirockets model and all requred parameter files
        responses:
            200:
                description: Format of a successful response. The model will be returned instead of just the name
                examples:
                    application/json: "{Model:MR,\nScaler:Scaler\n,sgdc:sgdc}"

        """
        return {"Model":miniRocketsJson,
                "Scaler":miniRocketsJsonScaler,
                "sgdc":miniRocketsJsonSgdc}

# Convert the XGBoost model to JSON
class getXGBoost(Resource):
    def get(self):
        """
        Get the XGBoost model
        ---
        tags:
            - Get model
        description: Returns the XGBoost model and all requred parameter files
        responses:
            200:
                description: Format of a successful response. The model will be returned instead of just the name
                examples:
                    application/json: "{Model:xgb,\nScaler:Scaler,\nFeatureNames:FeatureNames}"

        """
        return {"Model":XGBoostJson,
                "Scaler":XGBoostJsonScaler,
                "FeatureNames":xgb_featnames}

class getXGBoostPred(Resource):
    def get(self, partition):
        """
        Get the XGBoost model prediction data
        ---
        tags:
            - Get prediction data
        description: Returns the prediction data calculated by the XGBoost model for the selected partition
        parameters: [{
            "name": "partition",
            "description": "The desired partition. Will return nothing on an empty or invalid input\n
            Valid inputs: partition1, partition2, partition3, partition4, partition5",
            "allowMultiple": False,
            "in": path,
            "type": "string"}]
        responses:
            200:
                description: Format of a successful response. The model will be returned instead of just the name
                examples:
                    application/json: "{file_name:{FL:0.0, NF:1.0}}"

        """
        data = {}
        if partition:
            if(str(partition) == "partition1"):
                with open(os.path.join(models_dir, "prediction_data", "XGBoostPredPart1.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition2"):
                with open(os.path.join(models_dir, "prediction_data", "XGBoostPredPart2.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition3"):
                with open(os.path.join(models_dir, "prediction_data", "XGBoostPredPart3.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition4"):
                with open(os.path.join(models_dir, "prediction_data", "XGBoostPredPart4.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition5"):
                with open(os.path.join(models_dir, "prediction_data", "XGBoostPredPart5.json"), "rb") as f:
                    data = json.load(f)
        return data

class getLSTMPred(Resource):
    def get(self, partition):
        """
        Get the LSTM model prediction data
        ---
        tags:
            - Get prediction data
        description: Returns the prediction data calculated by the LSTM model for the selected partition
        parameters: [{
            "name": "partition",
            "description": "The desired partition. Will return nothing on an empty or invalid input\n
            Valid inputs: partition1, partition2, partition3, partition4, partition5",
            "allowMultiple": False,
            "in": path,
            "type": "string"}]
        responses:
            200:
                description: Format of a successful response. The model will be returned instead of just the name
                examples:
                    application/json: "{file_name:{FL:0.0, NF:1.0}}"

        """
        data = {}
        if partition:
            if(str(partition) == "partition1"):
                with open(os.path.join(models_dir, "prediction_data", "LSTMPredPart1.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition2"):
                with open(os.path.join(models_dir, "prediction_data", "LSTMPredPart2.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition3"):
                with open(os.path.join(models_dir, "prediction_data", "LSTMPredPart3.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition4"):
                with open(os.path.join(models_dir, "prediction_data", "LSTMPredPart4.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition5"):
                with open(os.path.join(models_dir, "prediction_data", "LSTMPredPart5.json"), "rb") as f:
                    data = json.load(f)
        return data
        
class getMiniRocketsPred(Resource):
    def get(self, partition):
        """
        Get the MiniRockets model prediction data
        ---
        tags:
            - Get prediction data
        description: Returns the prediction data calculated by the MiniRockets model for the selected partition
        parameters: [{
            "name": "partition",
            "description": "The desired partition. Will return nothing on an empty or invalid input\n
            Valid inputs: partition1, partition2, partition3, partition4, partition5",
            "allowMultiple": False,
            "in": path,
            "type": "string"}]
        responses:
            200:
                description: Format of a successful response. The model will be returned instead of just the name
                examples:
                    application/json: "{file_name:{FL:0.0, NF:1.0}}"

        """
        data = {}
        if partition:
            if(str(partition) == "partition1"):
                with open(os.path.join(models_dir, "prediction_data", "MiniRocketsPredPart1.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition2"):
                with open(os.path.join(models_dir, "prediction_data", "MiniRocketsPredPart2.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition3"):
                with open(os.path.join(models_dir, "prediction_data", "MiniRocketsPredPart3.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition4"):
                with open(os.path.join(models_dir, "prediction_data", "MiniRocketsPredPart4.json"), "rb") as f:
                    data = json.load(f)
            elif(str(partition) == "partition5"):
                with open(os.path.join(models_dir, "prediction_data", "MiniRocketsPredPart5.json"), "rb") as f:
                    data = json.load(f)
        return data
# Get predictions- JSON containing full prediction data
#   Add functions to predict over whole dataset using loaded models, run and store while loading
#   Use similar prediction method found in each models testing evaluation, and run over whole set
#   Could also do as individual notebook, store file to return
# Predict- Take in required list of parameters, and return likelihood of a flare/flare category
#   May not be viable for all models, such as LSTM (requires time series, too many variables for reasonable use)
#   Check inputs for predictions, create for single sample predictions (may also need to standardise input before predicting)


# add endpoints to app
api.add_resource(getLightGBM,"/api/get/LightBGM/<feature>")
api.add_resource(getLSTM,"/api/get/LSTM")
api.add_resource(getMiniRockets,"/api/get/minirockets")
api.add_resource(getXGBoost,"/api/get/xgboost")
api.add_resource(getXGBoostPred,"/api/getPredictions/XGBoost/<partition>")
api.add_resource(getLSTMPred,"/api/getPredictions/LSTM/<partition>")
api.add_resource(getMiniRocketsPred,"/api/getPredictions/MiniRockets/<partition>")