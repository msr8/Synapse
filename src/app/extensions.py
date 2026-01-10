from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from flask_socketio import SocketIO
from sty import fg, bg, ef, rs


print(ef.bold + fg.green + 'Loading extensions...' + rs.all)

db   = SQLAlchemy()
sess = Session()
sock = SocketIO()
