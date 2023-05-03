import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels as smodel
import statsmodels.api as sm

from custom_logger import Logger

# Constants
BASE_DIR = "./"
TRAIN_DATA_FILENAME = "exercise_26_train.csv"
TRAINED_MODEL_FILENAME = "results.pkl"

log = Logger()

class GLMModel():
    def __init__(self, model_file: str=""):
        log.info("Instantiating GLMModel")
        self.raw_train = self.load_data(TRAIN_DATA_FILENAME)
        self.imputer = None
        self.std_scaler = None
        self.all_train = None
        self.variables = self.get_variables()
        if model_file == "" or not os.path.isfile(os.path.join(BASE_DIR, model_file)):
             self.model = self.build_model()
        else:
            self.model = sm.load(model_file)
        self.bins = self.get_bins()

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Get csv file and load into dataframe
        """
        log.info("[load_data] called")
        try:
            if os.path.isfile(os.path.join(BASE_DIR, filename)):
                df = pd.read_csv(filename)
                return df
            raise Exception(f"[load_data] filename({filename}) not found")

        except Exception as e:
            log.error("[load_data] error: "+str(e))
            return None

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace money values and percentages so field values are floats
        """
        log.info("[feature_engineering] called")
        try:
            data = df.copy(deep=True)

            # Replace money variables and percents as floats
            data['x12'] = data['x12'].str.replace('$','')
            data['x12'] = data['x12'].str.replace(',','')
            data['x12'] = data['x12'].str.replace(')','')
            data['x12'] = data['x12'].str.replace('(','-')
            data['x12'] = data['x12'].astype(float)
            data['x63'] = data['x63'].str.replace('%','')
            data['x63'] = data['x63'].astype(float)

            return data
        except Exception as e:
            log.error("[feature_engineering] error: "+str(e))
            return pd.DataFrame([])

    def data_prepping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in missing values and rename string values as columns
        """
        log.info("[data_prepping] called")
        try:
            data = df.copy(deep=True)

            # Impute all rows of numerical values to replace missing values with means of that column and standardize
            if self.imputer is None:
                self.imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
                data_imputed = pd.DataFrame(self.imputer.fit_transform(data.drop(columns=['y','x5', 'x31',  'x81' ,'x82'])), columns=data.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
            else:
                data_imputed = pd.DataFrame(self.imputer.transform(data.drop(columns=['y','x5', 'x31',  'x81' ,'x82'])), columns=data.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
            
            if self.std_scaler is None:
                self.std_scaler = StandardScaler()
                data_imputed_std = pd.DataFrame(self.std_scaler.fit_transform(data_imputed), columns=data_imputed.columns)
            else:
                data_imputed_std = pd.DataFrame(self.std_scaler.transform(data_imputed), columns=data_imputed.columns)

            # Replace all rows with string values to an additional row at the end and 1 under the column that had that string value, 0 otherwise
            dumb5 = pd.get_dummies(data['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
            data_imputed_std = pd.concat([data_imputed_std, dumb5], axis=1, sort=False)

            dumb31 = pd.get_dummies(data['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
            data_imputed_std = pd.concat([data_imputed_std, dumb31], axis=1, sort=False)

            dumb81 = pd.get_dummies(data['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
            data_imputed_std = pd.concat([data_imputed_std, dumb81], axis=1, sort=False)

            dumb82 = pd.get_dummies(data['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
            data_imputed_std = pd.concat([data_imputed_std, dumb82], axis=1, sort=False)
            data_imputed_std = pd.concat([data_imputed_std, data['y']], axis=1, sort=False)

            del dumb5, dumb31, dumb81, dumb82

            return data_imputed_std
        except Exception as e:
            log.error("[data_prepping] error: "+str(e))
            return pd.DataFrame([])

    def api_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean values from api data and impute with raw training data
        """
        log.info("[api_data_cleaning] called")
        try:
            data = df.copy(deep=True)
            index = self.raw_train.shape[0]
            # Concat with train data to ensure all columns are created
            df_full = pd.concat([self.raw_train, data], sort=False).reset_index(drop=True)
            data_val = self.feature_engineering(df_full)
            data_prepped = self.data_prepping(data_val)
            # Return only the rows from the intial dataframe and none of the training rows
            return data_prepped.loc[index:]
        except Exception as e:
            log.error("[api_data_cleaning] error: "+str(e))
            return pd.DataFrame([])

    def get_variables(self) -> list:
        """
        Get variables used in modeling
        """
        log.info("[get_variables] called")
        try: 
            train_val = self.feature_engineering(self.raw_train)

            x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

            train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
            val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
            test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

            train_imputed_std = self.data_prepping(train)
            val_imputed_std = self.data_prepping(val)
            test_imputed_std = self.data_prepping(test)

            # Recombine all training data now that it is imputed and standardized
            train_and_val = pd.concat([train_imputed_std, val_imputed_std])
            self.all_train = pd.concat([train_and_val, test_imputed_std])

            # Find variables to use in finding phat
            exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
            exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
            exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
            exploratory_results['coefs'] = exploratory_LR.coef_[0]
            exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
            var_reduced = exploratory_results.nlargest(25,'coefs_squared')
            variables = var_reduced['name'].to_list()
            return variables
        except Exception as e:
            log.error("[get_variables] error: "+str(e))
            return None

    def build_model(self) -> smodel.discrete.discrete_model.BinaryResultsWrapper:
        """
        Create model from training data
        """
        log.info("[build_model] called")
        try:
            # Create prediction model and save model to a pickle file
            final_logit = sm.Logit(self.all_train['y'], self.all_train[self.variables])
            final_result = final_logit.fit()
            final_result.save(TRAINED_MODEL_FILENAME)
            return final_result
        except Exception as e:
            log.error("[build_model] error: "+str(e))
            return None

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return probability of a given row of features
        """
        log.info("[predict] called")
        try:
            # Predict on the data given and return series
            return pd.DataFrame(self.model.predict(df[self.variables])).rename(columns={0:'phat'})
        except Exception as e:
            log.error("[predict] error: "+str(e))
            return None

    def get_bins(self) -> pd.DataFrame:
        """
        Get the bins for prediction outcomes
        """
        log.info("[get_bins] called")
        try:
            Outcomes_train_final = self.predict(self.all_train)
            Outcomes_train_final['y'] = self.all_train['y']
            Outcomes_train_final['prob_bin'] = pd.qcut(Outcomes_train_final['phat'], q=20)
            return Outcomes_train_final['prob_bin']
        except Exception as e:
            log.error("[get_bins] error: "+str(e))
            return pd.DataFrame([])