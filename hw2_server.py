'''

CSE535: Assignment-2
Testing service at http://10.218.107.121:5432/

'''

from flask import Flask, request
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from statistics import mode
import pickle
import json

app = Flask(__name__)
actions = {"buy": 1, "communicate":2, "fun": 3, "hope": 4, "mother":5,"really":6}


def convert_to_csv(json_data):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    data = json_data
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    # path_to_video = path_to_video[:-5]
    return pd.DataFrame(csv_data, columns=columns)


# Only required for training or it the request is a GET request
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
    df = df.drop(df.columns[[0, 1]], axis=1) # Removing the scores from the feature extraction
    train_x, test_x, train_y, test_y = train_test_split(df, y, test_size = 0.30, shuffle = True)
    return train_x, test_x, train_y, test_y


# prediction using Logistic Regression
def logistic_regression(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("logistic_regression_model.sav", 'rb'))
    lr_prediction = loaded_model.predict(test_x)
    # print("Accuracy for Logistic Regression Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(lr_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label


# prediction using a Decision Tree
def decision_tree(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("decision_tree_model.sav", 'rb'))
    decision_prediction = loaded_model.predict(test_x)
    # print("Accuracy for Decision Tree Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(decision_prediction)
    for k,v in actions.items():
        if v == result:
            label = k
    return label


# prediction using NN
def neural_network(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("neural_network_model.sav", 'rb'))
    nn_prediction = loaded_model.predict(test_x)
    # print("Accuracy for Neural Network Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(nn_prediction)
    for k,v in actions.items():
        if v == result:
            label = k
    return label


# prediction using SVM
# SVM Will take approximately 10-15 minutes, if training
def svm(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("svm_classifier_model.sav", 'rb'))
    svm_prediction = loaded_model.predict(test_x)
    # print("Accuracy for SVM = ", loaded_model.score(test_x, test_y)*100)
    result = mode(svm_prediction)
    for k,v in actions.items():
        if v == result:
            label = k
    return label


# GET request
@app.route('/')
def main_get():
    print("Received a GET request")
    df = pd.DataFrame()
    y = pd.DataFrame()
    train_x, test_x, train_y, test_y = load_data(df, y, actions)
    label_lr = logistic_regression(train_x, test_x, train_y, test_y)
    label_dt = decision_tree(train_x, test_x, train_y, test_y)
    label_nn = neural_network(train_x, test_x, train_y, test_y)
    label_svm = svm(train_x, test_x, train_y, test_y)
    
    result = {"1" : label_lr, 
              "2" : label_dt, 
              "3" : label_nn,
              "4" : label_svm} 
    print(result)
    result = json.dumps(result)
    return str(result)


# POST request
@app.route('/sendJsonData', methods=['GET', 'POST'])
def main_post():
    print("Received a POST request")
    content = request.get_json(force=True)
    test_x = None
    if (content != None):
        test_x = convert_to_csv(content)
        print(test_x)
        test_x = test_x.drop(test_x.columns[[0]], axis=1)  # Removing the scores from the feature extraction
        print(test_x.shape)
        # replace missing values with mean
        test_x.fillna(test_x.mean(), inplace=True)
    print("\n\n\n\n")
    train_x = None
    train_y = None
    test_y = None
    label_lr = str(logistic_regression(train_x, test_x, train_y, test_y))
    label_dt = str(decision_tree(train_x, test_x, train_y, test_y))
    label_nn = str(neural_network(train_x, test_x, train_y, test_y))
    label_svm = str(svm(train_x, test_x, train_y, test_y))
    
    result = {"1" : label_lr,
              "2" : label_dt, 
              "3" : label_nn,
              "4" : label_svm}
    result = json.dumps(result)
    print(result)
    return str(result)
