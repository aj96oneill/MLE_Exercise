import os
import pandas as pd
import sys
from unittest import mock

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from model_builder import GLMModel

df = pd.read_csv("exercise_26_test.csv")

class TestGLMModelInit():
    """
    This test class makes sure that GLM Model can instantiate corerctly and handle errors
    """
    @mock.patch('model_builder.smodel.discrete.discrete_model.BinaryResultsWrapper.save')
    def test_when_no_model_saved(self, mocked):
        mocked.return_value = None
        model = GLMModel()
        assert mocked.call_count == 1

    @mock.patch('model_builder.sm.load')
    def test_when_a_model_saved(self, mocked):
        mocked.return_value = None
        model = GLMModel("results.pkl")
        assert mocked.call_count == 1
    
    @mock.patch('model_builder.os.path.isfile')
    def test_when_no_train_data(self, mocked):
        mocked.return_value = False
        model = GLMModel("results.pkl")
        assert mocked.call_count > 0
        assert model.raw_train == None

class TestApiDataCleaning():
    """
    This test class tests that api_data_cleaning can be called and handle normal use case and can handle when data fails
    """
    def test_normal_case(self):
        test_data = df.loc[0:0]
        model = GLMModel("results.pkl")
        clean_data = model.api_data_cleaning(test_data)
        assert clean_data.shape[0] == 1
        assert clean_data.shape[1] == 122

    def test_when_data_is_bad(self):
        model = GLMModel("results.pkl")
        clean_data = model.api_data_cleaning([])
        assert clean_data.shape[0] == 0
        assert clean_data.shape[1] == 0

class TestRemainingErrorCases():
    """
    This test class tests that all methods in the class can handle an error
    """
    def test_feature_engineering(self):
        model = GLMModel("results.pkl")
        data = model.feature_engineering([])
        assert data.shape[0] == 0
        assert data.shape[1] == 0

    def test_data_prepping(self):
        model = GLMModel("results.pkl")
        data = model.data_prepping([])
        assert data.shape[0] == 0
        assert data.shape[1] == 0

    @mock.patch('model_builder.GLMModel.feature_engineering')
    def test_get_variables(self, mocked):
        mocked.return_value = None
        model = GLMModel("results.pkl")
        data = model.get_variables()
        assert data is None

    @mock.patch('model_builder.sm.Logit')
    def test_build_model(self, mocked):
        mocked.return_value = None
        model = GLMModel("results.pkl")
        data = model.build_model()
        assert data is None
        

    def test_predict(self):
        model = GLMModel("results.pkl")
        data = model.predict([])
        assert data is None

    @mock.patch('model_builder.GLMModel.predict')
    def test_get_bins(self, mocked):
        mocked.return_value = None
        model = GLMModel("results.pkl")
        data = model.get_bins()
        assert data.shape[0] == 0
        assert data.shape[1] == 0
