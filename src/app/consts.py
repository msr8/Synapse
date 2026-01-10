from os import environ
from dotenv import load_dotenv

load_dotenv()


UPLOADS_DIR          = 'uploads'
LOGIN_FAIL_REDIRECT  = 'auth.page_login'
POST_LOGIN_REDIRECT  = 'root.page_dashboard'
POST_LOGOUT_REDIRECT = 'root.page_index'


GOOGLE_CLIENT_ID     = environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = environ.get('GOOGLE_CLIENT_SECRET')
GEMINI_API_KEY       = environ.get('GEMINI_API_KEY')