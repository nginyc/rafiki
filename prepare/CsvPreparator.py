import abc
import os
import re
import logging
import urllib
import pandas as pd
import numpy as np

from .BasePreparator import BasePreparator

HTTP_PREFIX = '^https?://'
DATA_DL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

logger = logging.getLogger('atm')

class FileType(object):
    LOCAL = 'local'
    HTTP = 'http'

class CsvPreparator(BasePreparator):
    def __init__(self, csv_file_url, class_column, query_columns):
        '''
        Extracts dataset from CSV files
        Args:
            csv_file_url - local or http(s) URI to .CSV file
            class_column - name of label column
            query_columns - iterable of column names for queries 
        '''
        self._csv_file_url = csv_file_url
        self._class_column = class_column
        self._query_columns = query_columns

    def process_data(self, queries_data, labels_data=None):
        '''
        Args:
            queries_data - iterable of dicts {<column>: <value:int>} as queries
            labels_data - iterable of ints as labels (if labelled)
        '''
        X = np.array([
                [query[column] for column in self._query_columns] # Sort list by query columns
                for query in queries_data
            ])
        
        y = None
        if labels_data:
            y = np.array(labels_data) 

        return X, y

    def get_train_data(self):
        csv_file_path = download_data(self._csv_file_url)
        df = pd.read_csv(csv_file_path)
        queries_df = df[self._query_columns]
        labels_df = df[self._class_column]
        queries_data = [x.to_dict() for i, x in queries_df.iterrows()] 
        labels_data = labels_df.tolist()

        X, y = self.process_data(queries_data, labels_data)
        return X, y


def download_data(csv_file_url):
    """
    Download a set of train data and return the
    path to the local data.
    """
    local_train_path, train_type = get_local_data_path(csv_file_url)

    # if the data are not present locally, try to download them from the internet
    if not os.path.isfile(local_train_path):
        if train_type == FileType.HTTP:
            assert (download_file_http(csv_file_url) == local_train_path)

    return local_train_path


def get_local_data_path(data_path):
    """
    given a data path of the form http://...", return the local
    path where the file should be stored once it's downloaded.
    """
    if data_path is None:
        return None, None

    m = re.match(HTTP_PREFIX, data_path)
    if m:
        path = data_path[len(m.group()):].split('/')
        return os.path.join(DATA_DL_PATH, path[-1]), FileType.HTTP

    return data_path, FileType.LOCAL



def download_file_http(url, local_folder=DATA_DL_PATH):
    """ Download a file from a public URL and save it locally. """
    filename = url.split('/')[-1]
    if local_folder is not None:
        ensure_directory(local_folder)
        path = os.path.join(local_folder, filename)
    else:
        path = filename

    if os.path.isfile(path):
        logger.warning('file %s already exists!' % path)
        return path

    logger.debug('downloading data from %s...' % url)
    f = urllib.request.urlopen(url)
    data = f.read()
    with open(path, 'wb') as outfile:
        outfile.write(data)
    logger.info('file saved at %s' % path)

    return path


def ensure_directory(directory):
    """ Create directory if it doesn't exist. """
    if not os.path.exists(directory):
        os.makedirs(directory)
