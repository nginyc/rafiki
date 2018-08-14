import os

METHOD_CONFIG_MAP = {
    'logreg': 'logistic_regression.json',
    'svm': 'support_vector_machine.json',
    'sgd': 'stochastic_gradient_descent.json',
    'dt': 'decision_tree.json',
    'et': 'extra_trees.json',
    'rf': 'random_forest.json',
    'gnb': 'gaussian_naive_bayes.json',
    'mnb': 'multinomial_naive_bayes.json',
    'bnb': 'bernoulli_naive_bayes.json',
    'gp': 'gaussian_process.json',
    'pa': 'passive_aggressive.json',
    'knn': 'k_nearest_neighbors.json',
    'mlp': 'multi_layer_perceptron.json',
    'ada': 'adaboost.json',
    'one_layer_tf': 'single_hidden_layer_tensorflow_model.json'
}

METHOD_CONFIG_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

__all__ = ['METHODS_MAP', 'METHOD_CONFIG_FOLDER_PATH']