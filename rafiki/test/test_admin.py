from rafiki.client import Client
import pytest

'''

- [ ] Admins can ban users
- [ ] Admins can list all users
- [ ] Admins cannot ban other admins (only superadmins can do this)
- [ ] Admins cannot ban themselves
- [ ] Admins cannot create a user with an invalid type
- [ ] Add documentation on new user 

'''

class AuthActions(object):
    def loginSuperAdmin(self):
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='superadmin@rafiki', password='rafiki')
        return client

    def loginAdmin(self):
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='admin@rafiki', password='rafiki')
        return client

@pytest.fixture
def auth():
    return AuthActions()

def testAdminCanListAllUsers(auth):
    client = auth.loginAdmin()
    # List user
    users = client.get_users()
    print(users)
    assert isinstance(users,list)

def testAdminCanBanUsers(auth):
    client = auth.loginAdmin()
    client.ban_user('app_developer@rafiki')
    client.ban_user('admin@rafiki')
    users = client.get_users()
    for user in users:
        if user['email'] == 'app_developer@rafiki':
            assert user["banned_data"] != None
        if user['email'] == 'admin@rafiki':
            assert user["banned_data"] == None