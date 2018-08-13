import abc
import pickle
import os


class BaseMethod():
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        '''
          Refer to corresponding method in 
          http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, X):
        '''
          Refer to corresponding method in 
          http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        '''
        raise NotImplementedError()

    def predict_proba(self, X):
        '''
          Refer to corresponding method in 
          http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        '''
        raise NotImplementedError()

    @classmethod
    def Save(self, model, model_dir, model_id):
        '''
        Save the model to the `model_dir` directory, uniquely identified by `model_id` 
        Returns: model_file_path
        '''
        model_file_path = os.path.join(model_dir, str(model_id) + '.pickle')

        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        return model_file_path

    @classmethod
    def Load(self, model_dir, model_id):
        '''
        Loads the model in the `model_dir` directory, uniquely identified by `model_id` 
        Returns: model
        '''
        model_file_path = os.path.join(model_dir, str(model_id) + '.pickle')

        with open(model_file_path, 'rb') as f:
            return pickle.load(f)
