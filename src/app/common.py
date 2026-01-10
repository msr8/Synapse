from collections import defaultdict

# task_id -> GenerativeModel object
chat_sessions = {}
# task_id -> idk
bayesian_results = defaultdict(list)