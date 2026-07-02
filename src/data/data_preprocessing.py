import numpy as np
import pandas as pd

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# fetch the data from data/raw
def load_data(train_path, test_path):
    train_data=pd.read_csv(train_path)
    test_data=pd.read_csv(test_path)
    return train_data, test_data


# transfrom the data
nltk.download('wordnet')
nltk.download('stopwords')

def lematization(text):
    lemmatizer=WordNetLemmatizer()
    text=text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words=set(stopwords.words('english'))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text=text.split()
    text=[y.lower() for y in text]
    return " ".join(text)
    
def removing_punctuations(text):
    punctuations =  r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    # raw string avoids invalid escape warnings
    text= re.sub('[%s]' % re.escape(punctuations), '', text)

    # remove extra whitespace
    text=re.sub(r'\s+', ' ', text)
    text=' '.join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern=re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentence(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split())<3:
            df.text.iloc[i]=np.nan

def normalize_text(df):
    df.Text=df.Text.apply(lambda Text: lower_case(Text))
    df.Text=df.Text.apply(lambda Text: remove_stop_words(Text))
    df.Text=df.Text.apply(lambda Text: removing_numbers(Text))
    df.Text=df.Text.apply(lambda Text: removing_punctuations(Text))
    df.Text=df.Text.apply(lambda Text: removing_urls(Text))
    df.Text=df.Text.apply(lambda Text: lematization(Text))
    return df




def save_data(data_path, train_processed_data, test_processed_data):
    os.makedirs(data_path, exist_ok=True)
    train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
    test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

def main():
    train_data, test_data = load_data('./emotion-detection/data/raw/train.csv', './emotion-detection/data/raw/test.csv')
    train_processed_data=normalize_text(train_data)
    test_processed_data=normalize_text(test_data)
    data_path = os.path.join('emotion-detection','data', 'interim')
    save_data(data_path, train_processed_data, test_processed_data)

if __name__=='__main__':
    main()