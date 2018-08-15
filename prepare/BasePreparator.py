import abc


class BasePreparator():
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass


    @abc.abstractmethod
    def process_data(self, queries_data, labels_data=None):
        '''
        Extracts (X, y) from raw queries & labels data configured for the preparator
        Returns:
            X - n-d numpy array of floats as queries
            y - 1d numpy array of floats as labels 
        '''
        raise NotImplementedError()
    

    @abc.abstractmethod
    def get_train_data(self):
        '''
        Extracts (X, y) from the configured underlying dataset
        Returns:
            X - n-d numpy array of floats as training examples' queries
            y - 1d numpy array of floats as training examples' labels 
        '''
        raise NotImplementedError()

