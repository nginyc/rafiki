import bcrypt

def hash_password(password):
    password_hash = bcrypt.hashpw(password, bcrypt.gensalt())
    return password_hash

def if_hash_matches_password(password, password_hash):
    return bcrypt.checkpw(password, password_hash)
    