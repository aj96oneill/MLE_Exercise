import json

from flask import Flask, request, Response
import pandas as pd

from custom_logger import Logger
from model_builder import GLMModel

# Constants
BASE_DIR = "./"
TRAINED_MODEL_FILENAME = "results.pkl"

log = Logger()
app = Flask(__name__)

# Check if model/data needed is available
log.info("Checking for model")
loaded_model = GLMModel(TRAINED_MODEL_FILENAME)
log.info("Model ready for API")

@app.route('/predict', methods=['POST'])
def get_predictions():
    """Get Predictions
    Purpose: Return prediction(s) from GLM Model

    Input: Request body will have 1 to N rows of data in JSON format 

    Output: Response body will have N responses each with variables (business_outcome, phat, and model inputs) in JSON format 
    """
    # Try getting data from request and cleaning the data
    try:
        #read data to a pandas df
        if not request.is_json: raise Exception("Request body was not the expected JSON format")
        
        content = request.get_json()
        if len(content) == 0: raise Exception("Request had no data in list")
        
        if isinstance(content, str):
            content = json.loads(content)
        
        if isinstance(content, list):
            if len(content[0]) == 0: raise Exception("Request had no data in dict")
            df = pd.read_json(json.dumps(content), "records")
        elif isinstance(content, dict):
            df=pd.DataFrame({k: [v] for k, v in content.items()})
        else:
            raise Exception("Content was not parsed correctly")
        
        # clean features
        clean_input_df = loaded_model.api_data_cleaning(df)
        if clean_input_df is None: raise Exception("Error while cleaning data, bad data given")

        # Get probabilities
        df_of_probs = loaded_model.predict(clean_input_df)
        if df_of_probs is None: raise Exception("Error while predicting")
        
        df_return = pd.concat([clean_input_df[loaded_model.variables], df_of_probs], axis=1, sort=False).reset_index(drop=True)

        # get class label based on buisness outcome
        bins_df = loaded_model.bins        
        threshold = bins_df.cat.categories[-5].left
        df_return['business_outcome'] = df_return['phat'].apply(lambda x: int(x > threshold))
        
        # Return in alphabetical order in JSON format
        df_return_ordered = df_return.reindex(sorted(df_return.columns), axis=1)
        responses_ordered = df_return_ordered.to_dict("records")
        return Response(json.dumps(responses_ordered), status=200, mimetype='application/json')
    
    except Exception as e:
        log.error(f"[get_predictions] has error: "+str(e))
        return Response(json.dumps({"error":str(e)}), status=400, mimetype='application/json')