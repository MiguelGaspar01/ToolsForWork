from flask import Flask, request, jsonify
import joblib
import pandas as pd


#creating the flask app

app = Flask(__name__) 

model = joblib.load("insert model path")
col_names = joblib.load("column_names-pk")

#connect post api call on the predict() function 

@app.route('/predict', methods=["POST"])  #execute the predict function

def predict():

    feat_data = request.json

    df = pd.Dataframe(feat_data)
    df = df.reindex(columns = col_names) #making sure columns match

    prediction = list(model.predict(df)) #execute the prediction

    return jsonify({"prediction": str(prediction)})



if __name__ == "__main__":

    app.run(debug=True)