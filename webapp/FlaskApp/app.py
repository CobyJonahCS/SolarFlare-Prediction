import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #This line was temp fix for some weird conda thing I dont remember

from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_cors import CORS
from flask_restful import Resource, Api
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from flasgger import Swagger, swag_from

#Directories for the models
base_dir = os.path.dirname(os.path.abspath(__file__))

models_dir = os.path.abspath(
    os.path.join(base_dir,"..", "..", "models")
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
        self.linear = nn.Linear(24, 2) # Errors out unless changed to 24,3 for me (Brooklyn). This change prevents LSTM prediction. TODO- Find cause of error

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


# TODO- API Doc work below here

# API methods
# Prediction calls- Cannot pass in CSV directly, need to check how each model handels predictions, pass in required parameters
# Get models- Return model information and weights, possible just the pickled forms
# get model parameters- Similar to above, but just required weights and info
# Extract predicition data- Could load in all data, then just run though prediction code. Select model, return prediction list with timestamps

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/xgboost.html")
def XGBoost():
    return render_template("xgboost.html")

@app.route("/LSTM.html")
def LSTM():
    return render_template("LSTM.html")

@app.route("/minirocket.html")
def miniRocket():
    return render_template("miniRocket.html")

@app.route("/Predictions.html")
def predictionsPage():
    return render_template("Predictions.html")

@app.route("/APIDocs.html")
def APIDocs():
    return render_template("APIDocs.html")

# API page management
app.config['SWAGGER'] = {
    'title': 'Solar Flare Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)
api = Api(app)

# Define API endpoints
class Test(Resource):
    def get(self,item=None):
        """
        This is an example endpoint that returns a given input
        ---
        tags:
            - Test
        description: Returns the input item
        parameters: [{
              "name": "item",
              "description": "String that will be returned",
              "allowMultiple": False,
              "in": path,
              "type": "string"}]
        responses:
            200:
                description: A successful response
                examples:
                    application/json: "{input: \\"test\\"}"

        """
        if item:
            return jsonify({"input":item})
        else:
            return jsonify([])

# Get models- JSON files containing model and parameters as found in models folder
#   Could re-use the loaded values from predictions
#   Need to check other branches for format of up-to-date models
# Get parameters- Similar to get models, but ignore main model file
#   May not be necessary, but wouldn't hurt to add
# Get predictions- JSON containing full prediction data
#   Add functions to predict over whole dataset using loaded models, run and store while loading
#   Use similar prediction method found in each models testing evaluation, and run over whole set
# Predict- Take in required list of parameters, and return likelihood of a flare/flare category
#   May not be viable for all models, such as LSTM (requires time series, too many variables for reasonable use)
#   Check inputs for predictions, create for single sample predictions (may also need to standardise input before predicting)

# Could group APIs, with user input for chosen model. Would result in cleaner docs, more work needed for swallow examples
# Group Get methods, keep predict separate, in case the required input parameters vary

# add endpoints to app
api.add_resource(Test,"/test/<item>")