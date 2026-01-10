from app.consts     import *
from app.models     import *
from app.extensions import db

from flask import request, session, redirect, url_for, flash, get_flashed_messages

from functools import wraps
from re import match
from bcrypt import hashpw, gensalt, checkpw



# Wrapper for login required but it also takes in a boolean indicating wether to return the data as json or not
def login_required(json=False):
    def wrapper(f):
        @wraps(f)
        def inner(*args, **kwargs):
            if not session.get('logged_in'):
                flash('You must be logged in to access this resource', 'error')
                # redirect_to = f.__module__.split('.')[-1] + '.' + f.__name__
                if json: return {'status': 'redirect', 'url': url_for(LOGIN_FAIL_REDIRECT)}, 401

                session['redirect_to'] = request.url
                return redirect(url_for(LOGIN_FAIL_REDIRECT))
            
            return f(*args, **kwargs)
        
        return inner
    return wrapper




def check_email(to_check:str) -> bool:
    return bool(match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', to_check))




def check_google_signup(email:str) -> bool:
    user = db.session.get(User, session['email'])    
    return user and (user.password is None)




def hash_stuff(text:str, salt:bool=False) -> str:
    our_salt = b'$2b$12$abcdefghijklmnopqrstuv' if not salt else gensalt()
    return hashpw(text.encode(), our_salt).decode()
