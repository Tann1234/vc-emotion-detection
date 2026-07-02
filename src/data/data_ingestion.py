import pandas as pd
import os
import numpy as np
import yaml
import logging


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# logging configure
logger= logging.getLogger('make_dataset')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('error.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)s -%(name)s-%(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        test_size = params["data_ingestion"]["test_size"]
        logger.debug('test_size retrieved')
        return float(test_size)
    except FileNotFoundError:
        logger.error('File not Found')
        raise 
    except KeyError as e:
        logger.error('yaml file is missing')
        raise 
    except ValueError:
        logger.error('some error occured')
        raise 
def read_data(data):
    try:
        df = pd.read_csv(data)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {data}")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file")

def process_data(df):
    try:
        final_df = df[(df['Emotion'] == 'sadness') | (df['Emotion'] == 'happy')]
        encoded = LabelEncoder()
        final_df['Emotion'] = encoded.fit_transform(final_df['Emotion'])
        return final_df
    except KeyError:
        raise KeyError("Column 'Emotion' not found in dataframe")

def save_data(data_path, train_data, test_data):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except PermissionError:
        raise PermissionError(f"Cannot write to {data_path}")

def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data('emotion-detection/src/data/Emotion_final.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join('emotion-detection','data', 'raw')
        save_data(data_path, train_data, test_data)
        print("Data ingestion completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
