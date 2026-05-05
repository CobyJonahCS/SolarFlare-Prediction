# Solar Flare Prediction
A web based application comparing multiple machine learning models for the prediction of solar flare intensity, based on the [SWAN-SF dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM). The webapp demonstrates comparisions between XGBoost, MiniRockets, and LSTM models, as well as a LightGBM forecaster.

Also available from the application are pages for predicting solar flare likelihood for input data in a chosen, LightGBM forecasting for given features (ABSNJZH, R_VALUE, TOTBSQ, TOTPOT, TOTUSJH, and TOTUSJZ), and an API to get the prediction data on the SWAN-SF dataset or extract the models for local running.

## Structure

- The basic html files can be fould within the [webapp folder](webapp), along with a lighter flask backend only containing the essential backend processed.
- The [FlaskApp folder](webapp/FlaskApp) contains the full flask hosted application. Steps to create and run this application can be found below.
- Data preprocessing and model creation steps can be found within the [notebooks folder](notebooks).
- Saved versions of the models can be found within [the models folder](models).

## To run
- Install the required dependancies as outlined in the [requirements](requirements.txt)
- To run the light backend, navigate to the [flask app folder](webapp/FlaskApp)
- Run the application using `python -m flask run`
- The web application can be found hosted locally on http://127.0.0.1:5000/