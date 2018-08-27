import abc
import random


class BasePreparator():
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass


    @abc.abstractmethod
    def transform_data(self, queries, labels=None):
        '''
        Extracts (X, y) from raw queries & labels data configured for the preparator
        Returns:
            X - numpy array of n-d numpy array of floats as queries
            y - numpy array of ints as labels 
        '''
        raise NotImplementedError()


    @abc.abstractmethod
    def reverse_transform_data(self, X=None, y=None):
        '''
        Converts (X, y) to raw queries & labels data configured for the preparator
        Returns: (queries, labels)
        '''
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_train_example(self, example_id=None):
        '''
        Returns a train example (x, y) from the configured underlying dataset, 
            optionally identified by example ID
        Returns:
            x - n-d numpy array of floats as the example's query
            y - int as the example's label 
        '''
        X, y = self.get_train_data()

        if example_id is not None:
            return list(zip(X, y))[example_id]
        
        return random.choice(list(zip(X, y)))

    @abc.abstractmethod
    def get_train_data(self):
        '''
        Extracts (X, y) from the configured underlying dataset
        Returns:
            X - numpy array of n-d numpy array of floats as training examples' queries
            y - numpy array of ints as training examples' labels 
        '''
        raise NotImplementedError()

