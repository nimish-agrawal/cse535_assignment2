# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import logging

from flask import Flask
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from statistics import mode
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle

app = Flask(__name__)
actions = {'buy': 1, 'communicate':2, 'fun': 3, 'hope': 4, 'mother':5,'really':6}


def load_data(df, y, actions):
    # Please change the path of the file accordingly
    data_path = "CSV/"
    for subdir, dirs, files in os.walk(data_path):
        action = subdir.split("/")[-1]
        if(action!=''):
            for file in files:
                df_temp = pd.read_csv(data_path+"/"+action+"/"+file,skiprows=1, header=None)
                df = pd.concat([df, df_temp])
                y_temp = pd.DataFrame([actions[action]]*df_temp.shape[0])
                y = pd.concat([y, y_temp])
    df = df.drop(df.columns[[1, 2]], axis=1) # Removing the scores from the feature extraction
    # if (df.isnull().values.any()) or (math.isnan(df)):
    #     df.fillna(df.mean())
    train_x, test_x, train_y, test_y = train_test_split(df, y, test_size = 0.30, shuffle = True)
    return train_x, test_x, train_y, test_y


def logistic_regression(train_x, test_x, train_y, test_y):
    global actions
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    lr_prediction = lr.predict(test_x)
    print("Accuracy for Logistic Regression Classifier = ", lr.score(test_x, test_y)*100)
    result = mode(lr_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label

def Decision_tree(train_x, test_x, train_y, test_y):
    decisiontree = DecisionTreeClassifier()
    decisiontree.fit(train_x, train_y)
    decision_prediction = decisiontree.predict(test_x)
    print("Accuracy for Decision Tree Classifier = ", decisiontree.score(test_x, test_y)*100)
    result = mode(decision_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label

def Neural_Network(train_x, test_x, train_y, test_y):
    NNclassifier = MLPClassifier(activation = 'relu', max_iter = 100, learning_rate = 'constant', batch_size = 'auto', 
                             hidden_layer_sizes = (25,), shuffle = True)
    NNclassifier.fit(train_x, train_y)
    nn_prediction = NNclassifier.predict(test_x)
    print(nn_prediction)
    print("Accuracy for Neural Network Classifier = ", NNclassifier.score(test_x, test_y)*100)
    result = mode(nn_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label

# SVM Will take approximately 10-15 minutes to give result. Please wait.
def SVM(train_x, test_x, train_y, test_y): 
    svm_classifier = svm.SVC(kernel = 'poly')
    svm_classifier.fit(train_x, train_y) 
    svm_prediction = svm_classifier.predict(test_x)
    print(svm_prediction)
    print("Accuracy for SVM = ", svm_classifier.score(test_x, test_y)*100)
    result = mode(svm_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label


@app.route('/')
def main():
    df = pd.DataFrame()
    y = pd.DataFrame()

    train_x, test_x, train_y, test_y = load_data(df, y, actions)

    label_lr = logistic_regression(train_x, test_x, train_y, test_y)
    label_dt = Decision_tree(train_x, test_x, train_y, test_y)
    label_nn = Neural_Network(train_x, test_x, train_y, test_y)
    label_svm = SVM(train_x, test_x, train_y, test_y)

    result = {"1": label_lr,
              "2": label_dt,
              "3": label_nn,
              "4": label_svm}
    print(result)
    return str(result)

'''
@app.route('/')
def hello_world():
    return 'Hello, World!'
'''

# @app.route('/', methods = ['POST'])
# def handle_request():
#     imagefile = flask.request.files['video']
#     print("Received Video :" + imagefile.filename)
#     # imagefile.save(filename)
#     return "Video uploaded Successfully!"
# @app.route('/', methods = ['GET'])
# def handle_requests():
#     return "Flask Server & Android are Working Well!"
# app.run(host = "0.0.0.0", port =5000,debug = True)

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
# [END app]
