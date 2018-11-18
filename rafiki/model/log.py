import os
import traceback
import datetime

class DuplicatePlotException(Exception): pass

class ModelLogUtils():
    '''
    Collection of utility methods for logging and plotting of messages & metrics during training.
    '''   
    def __init__(self):
        # Add logging to stdout for local debugging
        self._logger = ModelLogUtilsLogger()

    def set_logger(self, logger):
        if not isinstance(logger, ModelLogUtilsLogger):
            raise Exception('`logger` should subclass `ModelLogUtilsLogger`')
        
        self._logger = logger

    def log(self, message):
        '''
        Logs a message for analysis of model training.
        '''
        self._logger.log(message)

    def define_loss_plot(self):
        '''
        Convenience method of defining a plot of ``loss`` against ``epoch``.
        To be used with ``log_loss_metric()``.
        '''
        self.define_plot('Loss Over Epochs', ['loss'], x_axis='epoch')
  
    def log_loss_metric(self, loss, epoch):
        '''
        Convenience method for logging `loss` against `epoch`.
        To be used with ``define_loss_plot()``.
        '''
        self.log_metrics(loss=loss, epoch=epoch)

    def define_plot(self, title, metrics, x_axis=None):
        '''
        Defines a plot for a set of metrics for analysis of model training.
        By default, metrics will be plotted against time.
        '''
        self._logger.define_plot(title, metrics, x_axis)

    def log_metrics(self, **kwargs):
        '''
        Logs metrics for a single point in time { <metric>: <value> }.
        <value> should be a number.
        '''
        self._logger.log_metrics(**kwargs)

class ModelLogUtilsLogger():
    def __init__(self):
        self._plots = set()
    
    def log(self, message):
        self._print(message)

    def define_plot(self, title, metrics, x_axis):
        if title in self._plots:
            raise DuplicatePlotException('Plot {} already defined'.format(title))
        self._plots.add(title)
        self._print('Plot with title `{}` of {} against {} will be registered when this model is being trained on Rafiki' \
            .format(title, ', '.join(metrics), x_axis or 'time'))

    def log_metrics(self, **kwargs):
        self._print(', '.join(['{}={}'.format(metric, value) for (metric, value) in kwargs.items()]))

    def _print(self, message):
        print(message)