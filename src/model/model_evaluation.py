import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def load_model(model_path):
    clf=pickle.load(open(model_path, 'rb'))
    return clf

def load_data(data_path):
    test_data=pd.read_csv(data_path)
    X_test=test_data.iloc[:,0:-1].values
    y_test=test_data.iloc[:,-1].values
    return X_test, y_test



def make_prediction(clf,X_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    return y_pred, y_pred_proba

def metrics(y_test, y_pred, y_pred_proba):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    metrics_dict={
    'accuracy': accuracy, 
    'precision': precision,
    'recall': recall, 
    'auc': auc}
    return metrics_dict

def save_metrics(path,metrics_dict):
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def main():
    clf=load_model('models/model.pkl')
    X_test, y_test=load_data('./data/processed/test_bow.csv')
    y_pred, y_pred_proba=make_prediction(clf,X_test)
    metrics_dict=metrics(y_test, y_pred, y_pred_proba)
    save_metrics('reports/metrics.json', metrics_dict)



if __name__=='__main__':
    main()