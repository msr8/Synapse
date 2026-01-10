from app.extensions import db
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.dialects.postgresql import JSON
# https://www.digitalocean.com/community/tutorials/how-to-use-one-to-many-database-relationships-with-flask-sqlalchemy




class User(db.Model):
    __tablename__ = 'users'

    email     = db.Column(db.String(255), primary_key=True)
    username  = db.Column(db.String(255), nullable=False)
    password  = db.Column(db.String(60),  nullable=True) # Bcrypt "blowfish algo" salted hash (by default it is 60 characters long)
    signed_up = db.Column(db.DateTime,    nullable=False, server_default=db.func.now())
    tasks     = db.relationship('Task',   backref='owner', lazy=True)

    def __repr__(self):
        return f"User('{self.email}')"




class Usernames(db.Model):
    '''Exists only for O(1) lookup of usernames'''
    __tablename__ = 'usernames'

    username = db.Column(db.String(255), primary_key=True)
    # email    = db.Column(db.String(255), nullable=False)
    email    = db.Column(db.String(255), db.ForeignKey('users.email'), nullable=False)

    def __repr__(self):
        return f"Usernames('{self.username}')"




class Task(db.Model):
    __tablename__ = 'task'

    task_id                 = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name                    = db.Column(db.String(255),               nullable=False, default='Untitled Task')
    dataset                 = db.Column(db.Text,                      nullable=False) # CSV format
    processed_dataset       = db.Column(db.Text,                      nullable=True)  # CSV format
    processed_dataset_wo_fs = db.Column(db.Text,                      nullable=True)  # CSV format
    classes                 = db.Column(MutableList.as_mutable(JSON), nullable=False, default=[]) # dictionary, key: nominally encoded value, value: original value
    columns                 = db.Column(MutableList.as_mutable(JSON), nullable=False, default=[]) 
    target                  = db.Column(db.String(255),               nullable=True)
    insights                = db.Column(db.Text,                      nullable=True)
    messages                = db.Column(MutableList.as_mutable(JSON), nullable=False, default=[]) # Need MutableList.as_mutable() so that the changes can be tracked in mutable stuff, ie lists in this case
    created_at              = db.Column(db.DateTime,                  nullable=False, server_default=db.func.now())
    owner_email             = db.Column(db.String(255), db.ForeignKey('users.email'), nullable=False)
    # charts                  = db.relationship('Chart', backref='task', lazy=True)




# class Chart(db.Model):
#     __tablename__ = 'chart'

#     task_id    = db.Column(db.Integer, db.ForeignKey('task.task_id'), nullable=False, primary_key=True)
#     idx        = db.Column(db.Integer,  nullable=False, primary_key=True) # Column idx, used in frontend
#     col        = db.Column(db.Text,     nullable=False) # Column name
#     data       = db.Column(db.Text,     nullable=False) # In b64 format
#     created_at = db.Column(db.DateTime, nullable=False, server_default=db.func.now())

#     def __repr__(self):
#         return f"Chart('{self.col}')"



