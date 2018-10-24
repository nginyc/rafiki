import os
import traceback
import datetime

class ModelLogUtils():
    def __init__(self):
        self._loggers = []
        
        # Add logging to stdout for local debugging
        self.add_logger(ModelLogUtilsLogger())

    def add_logger(self, logger):
        if not isinstance(logger, ModelLogUtilsLogger):
            raise Exception('`logger` should subclass `ModelLogUtilsLogger`')
        
        self._loggers.append(logger)

    # Logs a message for model training analytics
    def log(self, message):
        for logger in self._loggers:
            logger.log(message)

    # Defines a plot for a set of metrics for model training analytics
    # By default, metrics will be plotted against time
    def describe_plot(self, title, metrics, x_axis=None):
        for logger in self._loggers:
            logger.describe_plot(title, metrics, x_axis)

    # Logs metrics for a single point in time { <metric>: <value> }
    # <value> is either a number or a boolean
    def log_metrics(self, **kwargs):
        for logger in self._loggers:
            logger.log_metrics( **kwargs)

class ModelLogUtilsLogger():
    def log(self, message):
        self._print(message)

    def describe_plot(self, title, metrics, x_axis):
        self._print('Plot with title `{}` of {} against {} will be registered when this model is being trained on Rafiki' \
            .format(title, ', '.join(metrics), x_axis or 'time'))

    def log_metrics(self, **kwargs):
        self._print(', '.join(['{}={}'.format(metric, value) for (metric, value) in kwargs.items()]))

    def _print(self, message):
        print('[{}] {}'.format(datetime.datetime.now(), message))