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
import requests
import pickle

app = Flask(__name__)
actions = {'buy': 1, 'communicate':2, 'fun': 3, 'hope': 4, 'mother':5,'really':6}


def load_data(df, y, actions):
    # Please change the path of the file accordingly
    data_path = "D:/cse535/CSV/"
    for subdir, dirs, files in os.walk(data_path):
        action = subdir.split("/")[-1]
        if(action!=''):
            for file in files:
                df_temp = pd.read_csv(data_path+"/"+action+"/"+file,skiprows=1, header=None)
                df = pd.concat([df, df_temp])
                y_temp = pd.DataFrame([actions[action]]*df_temp.shape[0])
                y = pd.concat([y, y_temp])
    df = df.drop(df.columns[[1, 2]], axis=1) # Removing the scores from the feature extraction
    train_x, test_x, train_y, test_y = train_test_split(df, y, test_size = 0.30, shuffle = True)
    return train_x, test_x, train_y, test_y

def logistic_regression(train_x, test_x, train_y, test_y):
    # lr = LogisticRegression()
    # lr.fit(train_x, train_y)
    # pickle.dump(lr, open("logistic_regression_model.sav", 'wb'))
    loaded_model = pickle.load(open("logistic_regression_model.sav", 'rb'))
    lr_prediction = loaded_model.predict(test_x)
    print("Accuracy for Logistic Regression Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(lr_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label

def Decision_tree(train_x, test_x, train_y, test_y):
    #decisiontree = DecisionTreeClassifier()
    #decisiontree.fit(train_x, train_y)
    #pickle.dump(decisiontree, open("decision_tree_model.sav", 'wb'))
    loaded_model = pickle.load(open("decision_tree_model.sav", 'rb'))
    decision_prediction = loaded_model.predict(test_x)
    print("Accuracy for Decision Tree Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(decision_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label

def Neural_Network(train_x, test_x, train_y, test_y):
    #NNclassifier = MLPClassifier(activation = 'relu', max_iter = 100, learning_rate = 'constant', batch_size = 'auto', 
                            # hidden_layer_sizes = (25,), shuffle = True)
    #NNclassifier.fit(train_x, train_y)
    #pickle.dump(NNclassifier, open("neural_network_model.sav", 'wb'))
    
    loaded_model = pickle.load(open("neural_network_model.sav", 'rb'))
    nn_prediction = loaded_model.predict(test_x)
    # print(nn_prediction)
    print("Accuracy for Neural Network Classifier = ", loaded_model.score(test_x, test_y)*100)
    result = mode(nn_prediction)
    for k,v in actions.items():
        if v==result:
            label = k
    return label

# SVM Will take approximately 10-15 minutes to give result. Please wait.
def SVM(train_x, test_x, train_y, test_y): 
    #print("inside svm")
    #svm_classifier = svm.SVC(kernel = 'poly')
    #svm_classifier.fit(train_x, train_y) 
    #pickle.dump(svm_classifier, open("svm_classifier_model.sav", 'wb'))
    loaded_model = pickle.load(open("svm_classifier_model.sav", 'rb'))
    svm_prediction = loaded_model.predict(test_x)
    # print(svm_prediction)
    print("Accuracy for SVM = ", loaded_model.score(test_x, test_y)*100)
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
    
    result = {"1" : label_lr, 
              "2" : label_dt, 
              "3" : label_nn,
              "4" : label_svm} 
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