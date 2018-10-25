import os
import logging
import tempfile
from datetime import datetime
import json
import traceback
import time

logger = logging.getLogger(__name__)

JOB_LOGGER_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

def configure_logging(process_name):
    # Configure all logging to a log file
    logs_folder_path = os.environ['LOGS_FOLDER_PATH']
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='{}/{}.log'.format(logs_folder_path, process_name))

class JobLogger():
    def __init__(self):
        self._log_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8')

    def define_plot(self, title, metrics, x_axis):
        self._log_line(has_time=False, type='PLOT', title=title, metrics=metrics, x_axis=x_axis)

    def log(self, message):
        self._log_line(type='MESSAGE', message=message)
        
    def log_metrics(self, **kwargs):
        self._log_line(type='METRICS', **kwargs)

    # Clears all logs (excluding plot definitions) before a specific time
    def clear_logs(self, datetimeBefore=None):
        if datetimeBefore is None:
            datetimeBefore = datetime.now()
        
        self._log_file.seek(0)
        new_log_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8')

        # Only copy over lines in new file that are not before `datetimeBefore`
        for line in self._log_file:
            (log_datetime, _) = self._parse_line(line)
            log_datetime = datetime.strptime(log_datetime, JOB_LOGGER_DATETIME_FORMAT)
            if log_datetime is None or log_datetime >= datetimeBefore:
                new_log_file.write(line)

        # Switch to new log file
        self._log_file.close()
        os.remove(self._log_file.name)
        self._log_file = new_log_file

    # Read all logs as bytes
    def export_logs(self):
        self._log_file.seek(0)
        logs_bytes = self._log_file.read().encode('utf-8')
        return logs_bytes

    def destroy(self):
        # Remove temporary internal log file
        self._log_file.close()
        os.remove(self._log_file.name)

    # Import and completely replace all logs
    def import_logs(self, logs_bytes):
        self._log_file.seek(0)
        self._log_file.write(logs_bytes.decode('utf-8'))
        self._log_file.truncate()

    '''
    Read logs as (plots, metrics, messages)

    plots: { title: Plot }
    Plot: { title, metrics, x_axis }
    metrics: { name: Metric }
    Metric: { name, values: [(Datetime, value)] }
    messages: [(Datetime, message)]
    Datetime: string (%Y-%m-%dT%H:%M:%S)
    '''
    def read_logs(self):
        self._log_file.seek(0)

        plots = {}
        metrics = {}
        messages = []
        for line in self._log_file:
            (log_datetime, log_dict) = self._parse_line(line)
            
            if 'type' not in log_dict:
                continue

            log_type = log_dict['type']
            del log_dict['type']

            if log_type == 'MESSAGE':
                messages.append((log_datetime, log_dict.get('message')))

            elif log_type == 'METRICS':
                for (name, value) in log_dict.items():
                    if name not in metrics:
                        metrics[name] = { 
                            'name': name,
                            'values': []
                        }
                    metrics[name]['values'].append((log_datetime, value))

            elif log_type == 'PLOT':
                title = log_dict['title']
                plots[title] = log_dict

        return (plots, metrics, messages)

    # Logs dictionary to temporary internal log file in JSON as line, appending current time
    def _log_line(self, has_time=True, **kwargs):
        if has_time:
            kwargs['time'] = datetime.now().strftime(JOB_LOGGER_DATETIME_FORMAT)
        self._log_file.write('{}\n'.format(json.dumps(kwargs)))

    # Parses a log line as (log_datetime, log_dict)
    def _parse_line(self, line):
        log = None
        try:
            log = json.loads(line)
        except: 
            logger.warn('Error while reading line in log: "{}"'.format(line))
            logger.warn(traceback.format_exc())
            return (None, {})

        log_datetime = None
        if 'time' in log:
            log_datetime = log['time']

        return (log_datetime, log)

def _test_job_logger_for_train_worker():
    l = JobLogger()
    l.define_plot('Model Loss', ['loss'], None)

    # Model is being trained
    l.log('START')
    time.sleep(1)
    l.log_metrics(loss=3.42, learning_rate=0.01)
    time.sleep(1)
    l.log_metrics(loss=3.21, learning_rate=0.01)
    time.sleep(1)
    l.log_metrics(loss=3.11)
    l.log('END')

    # At the end of training, logs are exported and saved
    logs_bytes = l.export_logs()
    assert isinstance(logs_bytes, bytes)
    l.destroy()

    # App developer checks on logs
    l2 = JobLogger()
    l2.import_logs(logs_bytes)
    (plots, metrics, messages) = l2.read_logs()
    l2.destroy()

    assert len(plots) == 1 and 'Model Loss' in plots
    assert plots['Model Loss'] == { 'title': 'Model Loss', 'metrics': ['loss'], 'x_axis': None }
    assert len(metrics) == 2 and 'loss' in metrics and 'learning_rate' in metrics
    assert [x[1] for x in metrics['loss'].get('values')] == [3.42, 3.21, 3.11]
    assert [isinstance(x[0], str) for x in metrics['learning_rate'].get('values')] == [True, True] 
    assert [x[1] for x in messages] == ['START', 'END']
    
def _test_job_logger_for_predictor():
    l = JobLogger()
    l.define_plot('Queries', ['queries'], None)

    l.log('UP')

    # Predictor receives queries
    time.sleep(1)
    queries = 0
    while queries < 3:
        l.log_metrics(query=True)
        queries += 1
    time.sleep(2)
    while queries < 23:
        l.log_metrics(query=True)
        queries += 1
    time.sleep(1)

    # Predictor's logs are exported and cleared periodically
    logs_bytes = l.export_logs()
    l.clear_logs()

    # App developer checks on this period's logs
    l2 = JobLogger()
    l2.import_logs(logs_bytes)
    (plots, metrics, messages) = l2.read_logs()
    l2.destroy()
    assert len(plots) == 1 and 'Queries' in plots
    assert len(metrics) == 1 and 'query' in metrics
    assert len(metrics['query'].get('values')) == 23
    assert [x[1] for x in messages] == ['UP']

    # Predictor receives more queries
    time.sleep(1)
    queries = 0
    while queries < 40:
        l.log_metrics(query=True)
        queries += 1
    time.sleep(2)
    while queries < 43:
        l.log_metrics(query=True)
        queries += 1
    
    l.log('KILLED')
    l.log('DOWN')

    # Predictor's logs are exported and cleared periodically
    logs_bytes = l.export_logs()
    l.clear_logs()
    # l.destroy()

    # App developer checks on this period's logs
    l2 = JobLogger()
    l2.import_logs(logs_bytes)
    (plots, metrics, messages) = l2.read_logs()
    l2.destroy()
    assert len(plots) == 1 and 'Queries' in plots
    assert len(metrics) == 1 and 'query' in metrics
    assert len(metrics['query'].get('values')) == 43
    assert [x[1] for x in messages] == ['KILLED', 'DOWN']

if __name__ == '__main__':
    print('Testing `JobLogger` for train worker...')
    _test_job_logger_for_train_worker()
    print('Testing `JobLogger` for predictor...')
    _test_job_logger_for_predictor()
    print('All tests pass!')
    