import pandas as pd
import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# Loading model
model = pickle.load(open('../model/model_rossmann.pkl', 'rb'))

# Initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    '''
    Predicts the target variable for a given test dataset using the Rossmann model.

    Parameters:
        None

    Returns:
        A JSON response containing the predictions for the test dataset.

    Raises:
        None
    '''
    
    test_json = request.get_json()
    
    if test_json: # There is data
        if isinstance(test_json, dict): # Unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # Multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
        # Instantiate Rossmann class
        pipeline = Rossmann()
        
        # Data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # Feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # Data preparation
        df3 = pipeline.data_preparation(df2)
        
        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__=='__main__':
    app.run('0.0.0.0')