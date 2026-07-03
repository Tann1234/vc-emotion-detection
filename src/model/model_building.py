import numpy as np
import pandas as pd
import pickle
import yaml

from sklearn.ensemble import GradientBoostingClassifier

def load_params(params_path):
    params=yaml.safe_load(open(params_path, 'r'))['model_building']
    return params

def load_data(train_path):
    train_data=pd.read_csv(train_path)
    X_train=train_data.iloc[:,0:-1].values
    y_train=train_data.iloc[:,-1].values
    return X_train, y_train

def train_model(X_train,y_train, params):
    clf=GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    clf.fit(X_train, y_train)
    return clf


def save_model(clf, model_path="models/model.pkl"):
    pickle.dump(clf, open(model_path, "wb"))


def main():
    params=load_params('params.yaml')
    
    X_train, y_train=load_data('./data/processed/train_tfidf.csv')
    clf=train_model(X_train, y_train,params)
    save_model(clf)


if __name__=='__main__':
    main()