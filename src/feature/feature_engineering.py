import numpy as np
import pandas as pd

import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

def load_params(params_path):
    max_features=yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
    return max_features

def load_data(train_path, test_path):
    train_data=pd.read_csv(train_path)
    test_data=pd.read_csv(test_path)
    return train_data, test_data


def bag_of_words(train_data, test_data):
    X_train=train_data['Text'].values
    y_train=train_data['Emotion'].values
    X_test=test_data['Text'].values
    y_test=test_data['Emotion'].values
    return X_train, y_train, X_test, y_test

# Apply Bag of words (CountVectorize)
def count_vectorize(X_train, y_train, y_test,X_test, max_features):
    vectorize=CountVectorizer(max_features=max_features)
    X_train_bow=vectorize.fit_transform(X_train)
    X_test_bow=vectorize.transform(X_test)
    train_df=pd.DataFrame(X_train_bow.toarray())
    train_df['label']=y_train
    test_df=pd.DataFrame(X_test_bow.toarray())
    test_df['label']=y_test
    return train_df, test_df

def save_data(data_path, train_df, test_df):
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)


def main():
    max_features=load_params('parmas.yaml')
    train_data, test_data=load_data('./data/interim/train_processed.csv', './data/interim/test_processed.csv')
    X_train, y_train, X_test, y_test=bag_of_words(train_data, test_data)
    train_df, test_df=count_vectorize(X_train, y_train, y_test,X_test, max_features)
    data_path = os.path.join('data', 'processed')

    save_data(data_path, train_df, test_df)

if __name__=='__main__':
    main()

