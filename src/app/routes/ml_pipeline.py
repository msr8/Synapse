from app.consts           import *
from app.common           import chat_sessions, bayesian_results
from app.models           import *
from app.utils            import login_required
from app.extensions       import db, sock
from app.ml.eda           import handle_column, corr_heatmap, mi_heatmap, pairplot_chart, confusion_matrix_chart
from app.ml.preprocessing import preprocess
from app.ml.chatbot       import generate_insights, generate_chat_session, generate_bayesian_insights, BAYESIAN_PROMPT
from app.ml.bayesian      import initialise_bayesian, bayesian_search, MODELS, MODELS_DNS

from flask import Blueprint, render_template, session, redirect, url_for, flash, request, send_from_directory
from flask_restful import Resource
from flask_socketio import emit, send
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from markdown import markdown
from pandas import read_csv
import pandas as pd
from json import dumps
from time import sleep
from copy import deepcopy
import os
from io import StringIO, BytesIO


ml_pipe_bp = Blueprint('ml_pipeline', __name__)

def convert_to_html(text):
    ret = markdown(text, extensions=['extra', 'smarty'])
    if ret.startswith('<p>'): ret = ret[3:]
    if ret.endswith('</p>'):  ret = ret[:-4]
    return ret





class UploadAPI(Resource):
    @login_required(json=True)
    def post(self):
        if 'file' not in request.files:            return {'message': 'No file part',            'status': 'error'}, 400
        
        csv_file = request.files['file']

        if csv_file.filename == '':                return {'message': 'No selected file',        'status': 'error'}, 400
        if not csv_file.filename.endswith('.csv'): return {'message': 'File must be a CSV file', 'status': 'error'}, 400

        df = pd.read_csv(csv_file)
        if not df.shape[0]: return {'message': 'Empty CSV file', 'status': 'error'}, 400

        # Save the task
        task = Task(owner_email=session['email'], columns=df.columns.tolist(), dataset=df.to_csv(index=False))
        db.session.add(task)
        db.session.commit()

        # Save the file
        # task_dir = os.path.join(UPLOADS_DIR, str(task.task_id))
        # os.makedirs(task_dir) # exist_ok = False cause THE FOLDER SHOULD NOT EXIST
        # csv_file.save(os.path.join(task_dir, 'dataset.csv'))

        # Redirect to the task page
        url = url_for('ml_pipeline.page_task', task_id=task.task_id)
        return {'status': 'redirect', 'url': url}





@ml_pipe_bp.route('/task/<int:task_id>')
@login_required()
def page_task(task_id):
    task = db.session.get(Task, task_id)

    if not task: return 'Task not found', 404
    # Check if the task belongs to the user
    if task.owner_email != session['email']: return 'This task does not belong to you >:(', 403

    # filepath = os.path.join(UPLOADS_DIR, str(task_id), 'dataset.csv')
    # if not os.path.exists(filepath): return 'Task dataset not found', 404

    return render_template('ml_pipeline/task.html', task=task, models_dns=MODELS_DNS)





class ChangeTasknameAPI(Resource):
    @login_required(json=True)
    def post(self):
        # Extract data
        new_taskname = request.get_json().get('new_taskname')
        task_id      = request.json.get('task_id')
        if not new_taskname: return {'message': 'New taskname not provided', 'status': 'error'}, 400

        # Get the task
        task = Task.query.get(task_id)

        # Check if the task exists or not
        if not task: return {'message': 'Task not found', 'status': 'error'}, 404
        # Check if the task belongs to the user
        if task.owner_email != session['email']: return {'message': 'This task does not belong to you >:(', 'status': 'error'}, 403
        # Check if the task name is already the same
        if task.name == new_taskname: return {'message': 'Task name is already the same', 'status': 'error'}, 400
        
        # Update the taskname
        task.name = new_taskname
        
        return {'status': 'success'}




class DeleteTaskAPI(Resource):
    @login_required(json=True)
    def post(self):
        task_id = int(request.json['task_id'])
        if not task_id: return {'message': 'No task ID provided', 'status': 'error'}, 400

        task = db.session.get(Task, task_id)
        if not task: return {'message': f'Task not found with ID {task_id}', 'status': 'error'}, 404
        if task.owner_email != session['email']: return {'message': 'This task does not belong to you >:(', 'status': 'error'}, 403

        db.session.delete(task)
        db.session.commit()
        chat_sessions.pop(task_id, None)
        bayesian_results.pop(task_id, None)

        return {'status': 'success'}





class SetTargetAPI(Resource):
    @login_required(json=True)
    def post(self):
        task_id = int(request.json['task_id'])
        target  = request.json['target']

        if not task_id: return {'message': 'No task ID provided', 'status': 'error'}, 400
        if not target:  return {'message': 'No target provided',  'status': 'error'}, 400

        task = db.session.get(Task, task_id)
        if not task: return {'message': f'Task not found with ID {task_id}', 'status': 'error'}, 404
        if task.owner_email != session['email']: return {'message': 'This task does not belong to you >:(', 'status': 'error'}, 403

        task.target = target
        db.session.commit()

        return {'status': 'success', 'target': target}





@sock.on('eda')
def sock_eda(data):
    task_id = int(data['task_id'])
    n_unique_threshold = int( data['args'].get('n_unique_threshold', 10) )
    
    task = db.session.get(Task, task_id)
    
    # Check if the task exists or not
    if not task: emit('eda_error', {'message': 'Task not found'}); return
    # Check if the task belongs to the user
    if task.owner_email != session['email']: emit('eda_error', {'message': 'This task does not belong to you >:('}); return
    # Check if the task has a target
    if not task.target: emit('eda_error', {'message': 'Target not set'}); return

    # # Remove any previous charts we have
    # charts = task.charts
    # for i in charts: db.session.delete(i)    
    # db.session.commit()
    
    # Read csv from task.dataset
    df = read_csv(StringIO(task.dataset))
    target = task.target
    count = 0
    for idx,col in enumerate(df.columns):
        data = handle_column(df, col, target, n_unique_threshold)
        # Emit the chart back
        data = {'col':col, 'idx':idx} | data
        emit('eda_chart', data)
        # # Save the chart in our db
        # db.session.add(Chart(task_id=task_id, idx=idx, col=col, data=chart.render_data_uri()))
        # db.session.commit()
        # # Increment the count
        # count += 1
    emit('eda_done', {'message': 'EDA done', 'count': count})





@sock.on('preprocess')
def sock_preprocess(data):
    task_id = int(data['task_id'])
    
    task = db.session.get(Task, task_id)
    # Check if the task exists or not
    if not task: emit('preprocess_error', {'message': 'Task not found'}); return
    # Check if the task belongs to the user
    if task.owner_email != session['email']: emit('preprocess_error', {'message': 'This task does not belong to you >:('}); return
    # CHeck if target is set
    if not task.target: emit('preprocess_error', {'message': 'Target not set'}); return
    # If the data has already been preprocessed, return done
    # if task.processed_dataset: emit('preprocess_done'); return

    # Convert the args to numerical values
    for i in ['mi_threshold', 'corr_threshold']: data['args'][i] = float(data['args'][i])
    for i in ['n_features']:                     data['args'][i] =   int(data['args'][i])

    # Read csv from task.dataset
    df = read_csv(StringIO(task.dataset))
    if df.shape[0] == 0: emit('preprocess_error', {'message': 'Empty dataset'}); return

    # Preprocess the dataset
    try: pp_df_wo_fs = preprocess(df, task.target, **data['args'])
    except Exception as e: return emit('preprocess_error', {'message': {str(e)}}); return
    # Save the processed dataset(s)
    task.processed_dataset       = df.to_csv(index=False)
    task.processed_dataset_wo_fs = pp_df_wo_fs.to_csv(index=False)
    db.session.commit()

    emit('preprocess_done')





@sock.on('preprocess_charts')
def sock_preprocess_charts(data):
    task_id = int(data['task_id'])
    
    task = db.session.get(Task, task_id)
    # Check if the task exists or not
    if not task: emit('preprocess_charts_error', {'message': 'Task not found'}); return
    # Check if the task belongs to the user
    if task.owner_email != session['email']: emit('preprocess_charts_error', {'message': 'This task does not belong to you >:('}); return
    # CHeck if target is set
    if not task.target: emit('preprocess_charts_error', {'message': 'Target not set'}); return
    # If the data has not been preprocessed, return an error
    if not task.processed_dataset_wo_fs: emit('preprocess_charts_error', {'message': 'Data has not been preprocessed'}); return

    # Read csv from task.processed_dataset
    orig_df = read_csv(StringIO(task.dataset))
    df      = read_csv(StringIO(task.processed_dataset_wo_fs))

    emit('correlation_matrix_chart', {'base64': corr_heatmap(df)})
    emit('mutual_info_chart',        {'base64': mi_heatmap(df, task.target)})
    emit('pairplot_chart',           {'base64': pairplot_chart(orig_df, task.target)})

    emit('preprocess_charts_done')





@sock.on('start_bayesian')
def sock_start_bayesian(data):
    task_id = int(data['task_id'])
    task = db.session.get(Task, task_id)

    # Check if the task exists or not
    if not task: emit('bayesian_error', {'message': 'Task not found'}); return
    # Check if the task belongs to the user
    if task.owner_email != session['email']: emit('bayesian_error', {'message': 'This task does not belong to you >:('}); return
    # CHeck if target is set
    if not task.target: emit('bayesian_error', {'message': 'Target not set'}); return
    # If the data has not been preprocessed, return an error
    if not task.processed_dataset: return emit('bayesian_error', {'message': 'Data has not been preprocessed'}); return

    # Convert some args to numerical values or stuff
    args = data['args']
    for i in ['test_size_ratio', 'deadline_time', 'threshold_value', 'delta_value']: args[i] = float(data['args'][i])
    for i in ['n_iter', 'cv']: data['args'][i] = int(args[i])
    args['cm_normalize'] = None if args['cm_normalize'] == 'none' else args['cm_normalize']

    df     = read_csv(StringIO(task.processed_dataset))
    target = task.target
    # emit("model_list", {'models': list(MODELS.keys())})

    classes = read_csv(StringIO(task.dataset))[target].unique()
    x_train, x_test, y_train, y_test, gen_stopper = initialise_bayesian(df, target, args['test_size_ratio'], True, args['stopper'], args['deadline_time'], args['threshold_value'], args['delta_value'])

    # print(bayesian_results)
    # to_save_bayesian = task_id not in bayesian_results
    bayesian_results[task_id] = []
    for model in MODELS.keys():
        try:
            result = bayesian_search(x_train, y_train, x_test, y_test, model, gen_stopper, args['scorer'], args['cv'], args['n_iter'])
            cm_b64 = confusion_matrix_chart(result['clf'], x_test, y_test, classes=classes, model_dn=MODELS_DNS[model], normalize=args['cm_normalize'])
            del result['clf'] # Cause its not JSON serializable (its a python object)
            emit('model_result', {'model': model, 'confusion_matrix_base64': cm_b64} | result)
            bayesian_results[task_id].append(result | {'model_clf': MODELS[model]['model'].__name__, 'model_dn': MODELS_DNS[model], 'scorer': args['scorer']})
        except Exception as e:
            emit('bayesian_model_error', {'message': str(e), 'model': model})

    # if to_save_bayesian:

    # If the chat has not been generated yet, dont do anything
    if task_id in chat_sessions:
        # Generating the bayesian insights
        bayesian_insights = generate_bayesian_insights(bayesian_results[task_id])
        chat = chat_sessions[task_id]
        chat.send_message(BAYESIAN_PROMPT.format(bayesian_insights=bayesian_insights))





class InitialiseChatbotAPI(Resource):
    @login_required(json=True)
    def post(self):
        task_id = int(request.json['task_id'])
        task = db.session.get(Task, task_id)
        if not task: return {'message': 'Task not found', 'status': 'error'}, 404
        if task.owner_email != session['email']: return {'message': 'This task does not belong to you >:(', 'status': 'error'}, 403
        # If the data has not been preprocessed, return an error
        # if not task.processed_dataset_wo_fs: return {'message': 'Data has not been preprocessed', 'status': 'error'}, 400

        messages = deepcopy(task.messages)

        # Get or generate the insights
        if not task.insights:
            df = read_csv(StringIO(task.dataset))
            task.insights = generate_insights(df, task.target)
            db.session.commit()

        # Make sure the chat session exists
        if not task_id in chat_sessions: chat_sessions[task_id] = generate_chat_session(task.insights, (None if task_id not in bayesian_results else bayesian_results[task_id]), deepcopy(messages))

        for msg in messages:
            msg['message'] = convert_to_html(msg['message'])
        
        return {'status': 'success', 'insights': task.insights, 'messages': messages}





class ChatbotChatAPI(Resource):
    @login_required(json=True)
    def post(self):
        args    = request.get_json()
        task_id = int(args.get('task_id'))
        text    = args.get('text')

        if not task_id: return {'message': 'No task ID provided', 'status': 'error'}, 400
        if not text:    return {'message': 'No message provided', 'status': 'error'}, 400

        task = db.session.get(Task, task_id)
        if not task: return {'message': 'Task not found', 'status': 'error'}, 404
        if task.owner_email != session['email']: return {'message': 'This task does not belong to you >:(', 'status': 'error'}, 403

        # If chat session is not present, initialise it
        if not task_id in chat_sessions:  chat_sessions[task_id] = generate_chat_session(task.insights, (None if task_id not in bayesian_results else bayesian_results[task_id]), deepcopy(task.messages))

        chat = chat_sessions[task_id]
        text = request.json['text']

        try: response = chat.send_message(text)
        except InternalServerError as e: # type: ignore
            print(e)
            return {'message': 'Internal Server Error. Please reload the page', 'status': 'error'}, 500

        task.messages.extend( [{'role':'user', 'message':text}, {'role':'bot', 'message':response.text}] ) # If mutable stuff not done in models.py : Reassigning the list so that commit saves it. `.append` doesnt work
        db.session.commit()

        response = convert_to_html(response.text)

        return {'status': 'success', 'response': response}





class ChatbotResetChatAPI(Resource):
    @login_required(json=True)
    def post(self):
        task_id = int(request.json['task_id'])
        task = db.session.get(Task, task_id)
        if not task: return {'message': 'Task not found', 'status': 'error'}, 404
        if task.owner_email != session['email']: return {'message': 'This task does not belong to you >:(', 'status': 'error'}, 403

        if task_id in chat_sessions: chat_sessions.pop(task_id)
        task.messages.clear()
        db.session.commit()

        return {'status': 'success'}