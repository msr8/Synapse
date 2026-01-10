from app.consts     import *
from app.models     import *
from app.utils      import login_required
from app.extensions import sock, db
from time           import sleep

from flask import Blueprint, render_template, session, redirect, url_for, flash, request, send_from_directory
from flask_restful import Resource
from flask_socketio import emit, send

from json import dumps
import os



root_bp = Blueprint('root', __name__)





# Favicon (https://tedboy.github.io/flask/patterns/favicon.html)
@root_bp.route('/favicon.ico')
def page_favicon():
    return send_from_directory('static', 'favicon-128.ico', mimetype='image/vnd.microsoft.icon')




@root_bp.route('/')
def page_index():
    return render_template('index.html')


@root_bp.route('/learn-more')
def learn_more():
    return render_template('learn-more.html')




# @root_bp.route('/test')
# def page_test():
#     return render_template('test.html')

class TestAPI(Resource):
    def get(self):
        # Return all the arguments passed in the URL
        return request.args

    def post(self):
        # Return all the arguments passed in the request form & body
        form_args = request.form
        json_args = request.get_json()

        return {'form': form_args, 'json': json_args}



# @root_bp.route('/session')
# def page_session():
#     return f'<pre>{dumps(session, indent=4)}\n\nSession ID: {session.sid}</pre>'
    




@root_bp.route('/dashboard')
@login_required() # ALWAYS remember to keep login_required() below the route decorator, otherwise it will not work (idk why)
def page_dashboard():
    user = db.session.get(User, session['email'])
    return render_template('dashboard.html', user=user)
