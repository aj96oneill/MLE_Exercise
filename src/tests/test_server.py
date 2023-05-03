import json
import os
import sys
from unittest import mock

import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from server import app

df = pd.read_csv("exercise_26_test.csv")

class TestGetPredictions():
    """
    route: /predict
    methods: POST
    """
    def test_normal_case_list(self):
        test_data = df.loc[0:0]
        response = app.test_client().post('/predict', json=test_data.to_json(orient="records"))
        res = json.loads(response.data.decode('utf-8'))
        print(res)
        assert response.status_code == 200
        assert type(res[0]) is dict
        assert type(res) is list

    def test_normal_case_list_all_test_data(self):
        test_data = df
        response = app.test_client().post('/predict', json=test_data.to_json(orient="records"))
        res = json.loads(response.data.decode('utf-8'))
        print(res)
        assert response.status_code == 200
        assert type(res[0]) is dict
        assert type(res) is list

    def test_normal_case_dict(self):
        test_data = df.loc[0:0]
        response = app.test_client().post('/predict', json=test_data.to_dict(orient="records")[0])
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 200
        assert type(res[0]) is dict
        assert type(res) is list

    def test_req_no_data_in_list(self):
        response = app.test_client().post('/predict', json=[])
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 400
        assert type(res) is dict
        assert res['error'] == "Request had no data in list"

    def test_req_no_data_in_dict(self):
        response = app.test_client().post('/predict', json=[{}])
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 400
        assert type(res) is dict
        assert res['error'] == "Request had no data in dict"

    def test_req_not_json(self):
        response = app.test_client().post('/predict', data="")
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 400
        assert type(res) is dict
        assert res['error'] == "Request body was not the expected JSON format"
    
    @mock.patch('server.isinstance')
    def test_req_bad_data_parsing(self, mocked):
        mocked.return_value = False
        response = app.test_client().post('/predict', json=[{"x100":"True"}])
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 400
        assert type(res) is dict
        assert res['error'] == "Content was not parsed correctly"

    @mock.patch('model_builder.GLMModel.api_data_cleaning')
    def test_req_bad_data_cleaning(self, mocked):
        mocked.return_value = None
        response = app.test_client().post('/predict', json=[{"x100":"True"}])
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 400
        assert type(res) is dict
        assert res['error'] == "Error while cleaning data, bad data given"

    def test_req_bad_data_predicting(self):
        response = app.test_client().post('/predict', json=[{"x1":"True"}])
        res = json.loads(response.data.decode('utf-8'))
        assert response.status_code == 400
        assert type(res) is dict
        assert res['error'] == "Error while predicting"