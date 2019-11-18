from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from statistics import mode
import math
# import requests
import pickle
import json

app = Flask(__name__)
actions = {'buy': 1, 'communicate':2, 'fun': 3, 'hope': 4, 'mother':5,'really':6}



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
    # data = json.loads(open(path_to_video, 'r').read())
    # print(json_data)
    data = json_data
    # print type(data)
    # print data
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
    return pd.DataFrame(csv_data, columns=columns)# .to_csv(path_to_video + 'key_points.csv', index_label='Frames#'



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
    # if (df.isnull().values.any()) or (math.isnan(df)):
    #     df.fillna(df.mean())
    train_x, test_x, train_y, test_y = train_test_split(df, y, test_size = 0.30, shuffle = True)
    return train_x, test_x, train_y, test_y


def logistic_regression(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("logistic_regression_model.sav", 'rb'))
    lr_prediction = loaded_model.predict(test_x)
    # print("Accuracy for Logistic Regression Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(lr_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label


def Decision_tree(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("decision_tree_model.sav", 'rb'))
    decision_prediction = loaded_model.predict(test_x)
    # print("Accuracy for Decision Tree Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(decision_prediction)
    for k,v in actions.items():
        if v == result:
            label = k
    return label


def Neural_Network(train_x, test_x, train_y, test_y):
    loaded_model = pickle.load(open("neural_network_model.sav", 'rb'))
    nn_prediction = loaded_model.predict(test_x)
    # print("Accuracy for Neural Network Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(nn_prediction)
    for k,v in actions.items():
        if v == result:
            label = k
    return label


# SVM Will take approximately 10-15 minutes to give result. Please wait.
def SVM(train_x, test_x, train_y, test_y):
    print("inside svm")
    loaded_model = pickle.load(open("svm_classifier_model.sav", 'rb'))
    svm_prediction = loaded_model.predict(test_x)
    # print("Accuracy for SVM = ", loaded_model.score(test_x, test_y)*100)
    result = mode(svm_prediction)
    for k,v in actions.items():
        if v == result:
            label = k
    return label


@app.route('/')
def main():
    print("got a get request")
    df = pd.DataFrame()
    y = pd.DataFrame()
    
    train_x, test_x, train_y, test_y = load_data(df, y, actions)
    
    label_lr = logistic_regression(train_x, test_x, train_y, test_y)
    label_dt = Decision_tree(train_x, test_x, train_y, test_y)
    label_nn = Neural_Network(train_x, test_x, train_y, test_y)
    label_svm = SVM(train_x, test_x, train_y, test_y)
    
    result = {"1" : label_lr, 
              "2" : label_dt, 
              "3" : label_nn,
              "4" : label_svm} 
    print(result)
    return str(result)


@app.route('/sendJsonData', methods=['GET', 'POST'])
def mainJson():    
    content = request.get_json(force=True)
    # content = request.data
    # print(content)
    test_x = None
    if(content != None):
        test_x = convert_to_csv(content)
        print(test_x)
        test_x = test_x.drop(test_x.columns[[0]], axis=1)  # Removing the scores from the feature extraction
    print("\n\n\n\n")
    print(test_x)
    df = pd.DataFrame()
    y = pd.DataFrame()

    # train_x, test_xx, train_y, test_y = load_data(df, y, actions)
    train_x = None
    train_y = None
    test_y = None
    label_lr = logistic_regression(train_x, test_x, train_y, test_y)
    label_dt = Decision_tree(train_x, test_x, train_y, test_y)
    label_nn = Neural_Network(train_x, test_x, train_y, test_y)
    label_svm = SVM(train_x, test_x, train_y, test_y)
    
    result = {"1" : label_lr, 
              "2" : label_dt, 
              "3" : label_nn,
              "4" : label_svm} 
    print(result)
    return result



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