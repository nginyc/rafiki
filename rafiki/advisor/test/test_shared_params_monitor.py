import pytest
import numpy as np
from datetime import datetime, timedelta

from rafiki.test.utils import global_setup
from rafiki.model import SharedParams
from rafiki.advisor.utils import SharedParamsMonitor

class TestCache():
    def test_get_local_recent(self):
        monitor = SharedParamsMonitor()

        # Populate params
        monitor.add_params('1', 0.7, time=(datetime.now() - timedelta(hours=1)), worker_id='3') # 1h ago (another worker)
        monitor.add_params('2', 0.2, time=datetime.now(), worker_id='1') # now
        monitor.add_params('3', 1, time=(datetime.now() - timedelta(minutes=2)), worker_id='1') # 2 min ago
        monitor.add_params('4', 0.1, time=datetime.now(), worker_id='2') # now (another worker)
        monitor.add_params('5', 0.1, time=(datetime.now() - timedelta(minutes=1)), worker_id='1') # 1 min ago

        assert monitor.get_params(SharedParams.LOCAL_RECENT, worker_id='1') == '2' # Should be most recent in worker 1

    def test_get_global_recent(self):
        monitor = SharedParamsMonitor()

        # Populate params
        monitor.add_params('1', 0.7, time=(datetime.now() - timedelta(hours=1)), worker_id='3') # 1h ago (another worker)
        monitor.add_params('2', 0.2, time=datetime.now(), worker_id='1') # now
        monitor.add_params('3', 1, time=(datetime.now() - timedelta(minutes=2)), worker_id='1') # 2 min ago
        monitor.add_params('4', 0.1, time=datetime.now(), worker_id='2') # now (another worker)
        monitor.add_params('5', 0.1, time=(datetime.now() - timedelta(minutes=1)), worker_id='1') # 1 min ago

        assert monitor.get_params(SharedParams.GLOBAL_RECENT, worker_id='3') == '4' # Should be most recent across all workers

    def test_get_local_best(self):
        monitor = SharedParamsMonitor()

        # Populate params
        monitor.add_params('1', 0.7, time=(datetime.now() - timedelta(hours=1)), worker_id='3') # 1h ago (another worker)
        monitor.add_params('2', 0.2, time=datetime.now(), worker_id='1') # now
        monitor.add_params('3', 1, time=(datetime.now() - timedelta(minutes=2)), worker_id='1') # 2 min ago
        monitor.add_params('4', 0.1, time=datetime.now(), worker_id='2') # now (another worker)
        monitor.add_params('5', 0.1, time=(datetime.now() - timedelta(minutes=1)), worker_id='1') # 1 min ago

        assert monitor.get_params(SharedParams.LOCAL_BEST, worker_id='1') == '3' # Should be best in worker 1

    def test_get_global_best(self):
        monitor = SharedParamsMonitor()

        # Populate params
        monitor.add_params('1', 0.7, time=(datetime.now() - timedelta(hours=1)), worker_id='3') # 1h ago (another worker)
        monitor.add_params('2', 0.2, time=datetime.now(), worker_id='1') # now
        monitor.add_params('3', 1, time=(datetime.now() - timedelta(minutes=2)), worker_id='1') # 2 min ago
        monitor.add_params('4', 0.1, time=datetime.now(), worker_id='2') # now (another worker)
        monitor.add_params('5', 0.1, time=(datetime.now() - timedelta(minutes=1)), worker_id='1') # 1 min ago

        assert monitor.get_params(SharedParams.GLOBAL_BEST, worker_id='2') == '3' # Should be best across all workers
