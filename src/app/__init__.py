from app.consts import *
from app.routes import *
from flask import Flask, url_for
from flask_restful import Api
from flask_dance.contrib.google import make_google_blueprint
from app.extensions import db, sess, sock
from app.config import Config

from sty import fg, bg, ef, rs

from os import environ, makedirs


def create_app():
    print(ef.bold + fg.green + 'Creating app...' + rs.all)
    

    app = Flask(__name__)  
    api = Api(app)  
    # Load config
    # app.config.from_object('app.config.Config')
    app.config.from_object(Config)


    # Initialize extensions
    db.init_app(app)
    sess.init_app(app)
    sock.init_app(app)
    with app.app_context(): db.create_all()
    

    # Google login
    environ['OAUTHLIB_RELAX_TOKEN_SCOPE']  = '1'
    environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1' # ONLY ON LOCAL ENV
    google_bp = make_google_blueprint(
        client_id     = GOOGLE_CLIENT_ID,
        client_secret = GOOGLE_CLIENT_SECRET,
        scope = ['email'],
        offline = True,
        redirect_to = 'auth.page_google_authorised'
    )
    app.register_blueprint(google_bp, url_prefix='/login')
    

    # Register blueprints
    app.register_blueprint(root_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(ml_pipe_bp)

    # Register API resources
    api.add_resource(TestAPI,               '/api/test')
    api.add_resource(LoginAPI,              '/api/auth/login')
    api.add_resource(SignupAPI,             '/api/auth/signup')
    api.add_resource(ChangeUsernameAPI,     '/api/auth/change-username')
    api.add_resource(ChangePasswordAPI,     '/api/auth/change-password')
    api.add_resource(UploadAPI,             '/api/upload')
    api.add_resource(SetTargetAPI,          '/api/task/set-target')
    api.add_resource(DeleteTaskAPI,         '/api/task/delete-task')
    api.add_resource(ChangeTasknameAPI,     '/api/task/change-taskname')
    api.add_resource(InitialiseChatbotAPI,  '/api/task/chatbot/initialise')
    api.add_resource(ChatbotChatAPI,        '/api/task/chatbot/chat')
    api.add_resource(ChatbotResetChatAPI,   '/api/task/chatbot/reset')


    # Initialize the uploads directory
    # makedirs(UPLOADS_DIR, exist_ok=True)


    return app



