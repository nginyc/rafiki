import pytest
import json

from rafiki.config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.advisor.app import app
from rafiki.model.knob import IntegerKnob, CategoricalKnob, FixedKnob, FloatKnob, serialize_knob_config
from rafiki.test.utils import global_setup
from rafiki.constants import AdvisorType

class TestApp():
    @pytest.fixture(scope='class', autouse=True)
    def client(self):
        '''
        Initializes the flask app and returns the client for it
        '''
        client = app.test_client()
        app.testing = True
        yield client

    def test_up(self, client):
        '''
        Should be up
        '''
        res = client.get('/')
        assert res.status_code == 200

    def test_login(self, client):
        '''
        Superadmin can login
        '''
        res = client.post(
            '/tokens', 
            content_type='application/json',
            data=json.dumps({
                'email': SUPERADMIN_EMAIL,
                'password': SUPERADMIN_PASSWORD
            })
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        token = data['token']
        assert isinstance(token, str)

    def test_create_advisor(self, client):
        '''
        Can create & get valid knobs from advisor
        '''
        auth_headers = self._login(client, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD)

        # Create advisor
        # Should return advisor ID
        knob_values = ['value1', 'value2']
        knob_config = {
            'knob': CategoricalKnob(knob_values)
        }
        knob_config_str = serialize_knob_config(knob_config)
        res = client.post(
            '/advisors', 
            headers=auth_headers,
            content_type='application/json',
            data=json.dumps({
                'knob_config_str': knob_config_str
            })
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        advisor_id = data['id']
        assert isinstance(advisor_id, str)

        # Get knobs from created advisor
        # Should get valid knobs based on knob config
        res = client.post(
            '/advisors/{}/propose'.format(advisor_id),
            headers=auth_headers
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        knobs = data['knobs']
        assert isinstance(knobs, dict)
        assert knobs['knob'] in knob_values 

        # Can feedback to advisor
        res = client.post(
            '/advisors/{}/feedback'.format(advisor_id),
            headers=auth_headers,
            content_type='application/json',
            data=json.dumps({
                'knobs': knobs,
                'score': 1
            })
        )
        assert res.status_code == 200

    def test_delete_advisor(self, client):
        '''
        Can create & get valid knobs from advisor
        '''
        auth_headers = self._login(client, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD)

        # Create advisor
        knob_config_str = serialize_knob_config({})
        res = client.post(
            '/advisors', 
            headers=auth_headers,
            content_type='application/json',
            data=json.dumps({
                'knob_config_str': knob_config_str
            })
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        advisor_id = data['id']
        assert isinstance(advisor_id, str)

        # Delete created advisor
        res = client.delete(
            '/advisors/{}'.format(advisor_id),
            headers=auth_headers
        )
        assert res.status_code == 200

        # Deleted advisor should not exist
        res = client.post(
            '/advisors/{}/propose'.format(advisor_id),
            headers=auth_headers
        )
        assert res.status_code == 400

    def test_create_advisor_of_type(self, client):
        '''
        Can create advisor of a specific type
        '''
        auth_headers = self._login(client, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD)

        # Create advisor of specific type
        knob_config = {}
        knob_config_str = serialize_knob_config(knob_config)
        res = client.post(
            '/advisors', 
            headers=auth_headers,
            content_type='application/json',
            data=json.dumps({
                'knob_config_str': knob_config_str,
                'advisor_type': AdvisorType.BTB_GP
            })
        )
        assert res.status_code == 200

    def _login(self, client, email, password):
        '''
        Logins as superadmin and returns required auth headers
        '''
        res = client.post('/tokens', 
            content_type='application/json',
            data=json.dumps({
                'email': email,
                'password': password
            })
        )
        data = json.loads(res.data)
        token = data['token']
        auth_headers = {
            'authorization': 'Bearer {}'.format(token)
        }
        return auth_headers





