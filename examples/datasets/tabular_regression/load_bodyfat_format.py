import pandas as pd
import json
import tempfile
import shutil
import os

from sklearn.model_selection import train_test_split


def load(dataset_url, out_train_dataset_path, out_test_dataset_path, out_meta_txt_path, \
        target=None, features=None):
    '''
        Loads and converts an tabular dataset of sklearn boston housing to the 
        DatasetType `TABULAR`.

        :param str dataset_url: URL to download the dataset CSV file
        :param str out_train_dataset_path: Path to save the output train dataset file
        :param str out_test_dataset_path: Path to save the output test dataset file
        :param str out_meta_csv_path: Path to save the output dataset metadata .TXT file
        :param str target: The name of the column to be predicted of the dataset
        :param List[str] features: The list of the names of the columns used as the features 
    '''

    print('Converting and writing datasets...')

    table_meta = {}
    if target != None:
        table_meta['target'] = target
        table_meta['features'] = features

    X = pd.read_csv(dataset_url)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)

    _write_meta_txt(table_meta, out_meta_txt_path)   
    print('Dataset metadata file is saved at {}'.format(out_meta_txt_path))

    _write_dataset(table_meta, X_train, out_train_dataset_path)
    print('Train dataset file is saved at {}'.format(out_train_dataset_path))

    _write_dataset(table_meta, X_test, out_test_dataset_path)
    print('Test dataset file is saved at {}'.format(out_test_dataset_path))


def _write_meta_txt(table_meta, out_meta_text_path):
    '''
        Writes the txt that contains the meta data of the table (target and features)

        :param dict[str, str] table_meta: A JSON that contains the target and features
    '''
    with open(out_meta_text_path, 'w') as outfile:  
        json.dump(table_meta, outfile)

def _write_dataset(table_meta, data, out_dataset_path):
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, out_dataset_path.split('/')[-1].split('.')[0])
        data.to_csv(csv_path + '.csv', index=False)
        table_meta_path = os.path.join(d, 'table_meta.txt')
        with open(table_meta_path, 'w') as outfile:  
            json.dump(table_meta, outfile)

    # Zip and export folder as dataset
        out_path = shutil.make_archive(out_dataset_path, 'zip', d)
        os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

if __name__ == '__main__':
    # Loads the bodyfat dataset as `TABULAR` DatasetType
    load(
        dataset_url = 'https://course1.winona.edu/bdeppa/Stat%20425/Data/bodyfat.csv',
        out_train_dataset_path='/Users/pro/Desktop/rafiki_fork/data/bodyfat_train.zip',
        out_test_dataset_path='/Users/pro/Desktop/rafiki_fork/data/bodyfat_test.zip',
        out_meta_txt_path='/Users/pro/Desktop/rafiki_fork/data/bodyfat_meta.txt',
        features=['density',
                  'age',
                  'weight',
                  'height',
                  'neck',
                  'chest',
                  'abdomen',
                  'hip',
                  'thigh',
                  'knee',
                  'ankle',
                  'biceps',
                  'forearm',
                  'wrist'],
        target='bodyfat'
    )