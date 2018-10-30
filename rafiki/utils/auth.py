from flask import request
import os
import jwt
from functools import wraps

from rafiki.constants import UserType
from rafiki.config import APP_SECRET

class UnauthorizedException(Exception): pass
class InvalidAuthorizationHeaderException(Exception): pass
    
def generate_token(payload):
    token = jwt.encode(payload, APP_SECRET, algorithm='HS256')
    return token.decode('utf-8')

def decode_token(token):
    payload = jwt.decode(token, APP_SECRET, algorithms=['HS256'])
    return payload

def auth(user_types=None):

    # Superadmin can do anything
    user_types = user_types.append(UserType.SUPERADMIN)

    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            auth_header = request.headers.get('authorization', None)
            token = extract_token_from_header(auth_header)
            auth = decode_token(token)

            if user_types and auth.get('user_type') not in user_types:
                raise UnauthorizedException()

            return f(auth, *args, **kwargs)

        return wrapped
    return decorator

def extract_token_from_header(header):
    if header is None:
        raise InvalidAuthorizationHeaderException()
    
    parts = header.split(' ')
    
    if len(parts) != 2:
        raise InvalidAuthorizationHeaderException()

    if parts[0] != 'Bearer':
        raise InvalidAuthorizationHeaderException()

    token = parts[1]
    return token