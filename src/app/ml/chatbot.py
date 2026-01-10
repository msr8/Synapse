from app.ml.preprocessing import handle_missing_values, nominal_encoding, scale
from app.consts import GEMINI_API_KEY

from sklearn.feature_selection import mutual_info_classif
import google.generativeai as genai

import pandas as pd
import numpy as np

from rich import inspect

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name        = 'gemini-1.5-flash-8b',
    generation_config = generation_config
)


# Event if you dont know the answer to a question, make an educated guess. Do NOT say "I don't know"
PROMPT = '''You are an AI assistant that helps users analyze and interpret datasets interactively. Below is a detailed statistical summary of a dataset, including basic information, data types, missing values, descriptive statistics, column-wise analysis, correlation matrix, and post-preprocessing insights.

Your task is to answer user queries about this dataset, providing data-driven insights, trends, and interpretations. You can:

- Explain statistical properties like mean, median, standard deviation, skewness, and kurtosis.
- Identify correlations and dependencies between features.
- Highlight important features based on correlation and mutual information with the target column.
- Explain the impact of preprocessing steps such as outlier detection, scaling, and encoding.
- Assist users in deriving business or scientific conclusions based on data trends.

When responding:

- Use simple, clear language with examples if needed.
- Do NOT use technical language or jargon. Assume that the person is a layman and not a technical person.
- Avoid making up data; only use the provided information.
- If the data lacks direct answers (e.g., "What is the future trend of this dataset?"), explain why prediction isn't possible from the given data alone.
- Even if you don't know the answer to a question, make an educated guess. Do NOT say "I don't know."
- Be confident with your answer, and this is very important. Do not hesistate with your answers by using phrases like "however ..."

Now, the user will ask questions about the dataset. Answer them using the following insights:


{insights}
'''

BAYESIAN_PROMPT = '''Given below are the results of Bayesian optimisation of various models on the dataset. This message is not a question and is not being sent by the user. Keep the below insights in mind if the user asks about the models or stuff like "best model", "best hyperparameters", etc. If users ask about the "best model", explain the model algorithm and also the various hyperparameters in detail

{bayesian_insights}
'''



def generate_chat_session(insights:str, bayesian_results:str, messages:list = None):
    prompt = PROMPT.format(insights=insights)
    if bayesian_results: prompt += '\n\n\n\n' + BAYESIAN_PROMPT.format(bayesian_insights=generate_bayesian_insights(bayesian_results))
    history = [
        {
            'role': 'user',
            'parts': [prompt]
        }
    ]
    if messages:
        # Convert the messages to the required format by google
        for msg in messages:
            if msg['role'] == 'bot': msg['role'] = 'model'
            msg['parts'] = [msg.pop('message')]
        history.extend(messages)
    chat = gemini_model.start_chat(history=history)

    return chat


def generate_bayesian_insights(bayesian_results:dict) -> str:
    insights = []
    for result in bayesian_results:
        insights.append(f"Name: {result['model_dn']}")
        insights.append(f"Classifier: {result['model_clf']}")
        insights.append(f"Best Training Score: {result['best_training_score']}")
        insights.append(f"Testing Score: {result['testing_score']}")
        insights.append(f"Best Parameters: {result['best_params']}")
        insights.append(f"Number of Iterations: {result['n_iter']}")
        insights.append(f"Time Taken: {result['time_taken']}")
        insights.append(f"Scoring Metric: {result['scorer']}")
        insights.append("")
    return '\n'.join(insights)



def get_n_outliers(series:pd.Series) -> int:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    n_outliers = len(series[(series < lower_bound) | (series > upper_bound)])
    
    return n_outliers

def mad(series:pd.Series) -> float:
    mean = series.mean()
    mad  = (series - mean).abs().mean()
    return mad





def generate_insights(df:pd.DataFrame, target:str):
    insights = []

    # Basic Information
    insights.append("# Basic Information #")
    insights.append(f"Number of rows: {df.shape[0]}")
    insights.append(f"Number of columns: {df.shape[1]}")
    insights.append(f"Column names: {', '.join(df.columns)}")
    insights.append(f"Target column: {target}")

    # Data Types
    insights.append("\n# Data Types #")
    insights.append(str(df.dtypes))

    # Missing Values
    insights.append("\n# Missing Values #")
    insights.append(str(df.isnull().sum()))
    
    # Descriptive Statistics
    insights.append("\n# Descriptive Statistics #")
    insights.append(str(df.describe(include='all')))


    # --------------- Column-wise Analysis ---------------
    insights.append("")
    for col in df.columns:
        insights.append(f"\n### Analysis of Column \"{col}\" ###")
        series = df[col]
        insights.append(f"Data Type: {series.dtype}")
        insights.append(f"Number of Missing Values: {series.isnull().sum()}")
        insights.append(f"Number of Unique Values: {series.nunique()}")
        # insights.append(f"Value Counts: {series.value_counts().to_dict()}")
        insights.append(f"Most Frequent Value(s): {series.mode().index.to_list()}, appearing {series.value_counts().max()} times")
        insights.append(f"Least Frequent Value(s): {series.value_counts().idxmin()}, appearing {series.value_counts().min()} times")
        # Numerical data
        if pd.api.types.is_numeric_dtype(series):
            insights.append(f"Mean: {series.mean()}")
            insights.append(f"Median: {series.median()}")
            insights.append(f"Minimum Value: {series.min()}")
            insights.append(f"Maximum Value: {series.max()}")
            insights.append(f"Range: {series.max() - series.min()}")
            insights.append(f"Q1: {series.quantile(0.25)}")
            insights.append(f"Q3: {series.quantile(0.75)}")
            insights.append(f"IQR: {series.quantile(0.75) - series.quantile(0.25)}")
            insights.append(f"Standard Deviation: {series.std()}")
            insights.append(f"Variance: {series.var()}")
            insights.append(f"Skewness: {series.skew()}")
            insights.append(f"Kurtosis: {series.kurt()}")
            insights.append(f"MAD (Mean Absolute Deviation): {mad(series)}")
        # Time data
        elif pd.api.types.is_datetime64_any_dtype(series):
            insights.append(f"Start Date: {series.min()}")
            insights.append(f"End Date: {series.max()}")
            insights.append(f"Time Span: {series.max() - series.min()}")
        # print(col)
        # if col == 'Bare Nuclei': print('\n'.join(insights))



    # --------------- Post-Preprocessing Analysis ---------------
    pre_df = df.copy()
    handle_missing_values(pre_df, target)
    nominal_encoding(pre_df)
    scale(pre_df, target)
    x    = pre_df.drop(target, axis=1)
    y    = pre_df[target]
    corr = pre_df.corr()
    mi   = mutual_info_classif(x, y.astype('str')) # Returns an array ; `astype('str')` cause MI doesn't work with continuous data
    mi   = dict(zip(x.columns,mi))
    insights.append("\n\n\n\n# Post-Preprocessing Dataset Analysis #")
    insights.append(f"Correlation Matrix: \n{corr}")
    insights.append(f"Mutual Information with target column \"{target}\": {mi}")

    insights.append("\n\n# Post-Preprocessing Column Analysis #")
    for col in pre_df.columns:
        insights.append(f"\n### Post-Preprocessing Analysis of Column \"{col}\" ###")
        series = pre_df[col]
        n_outliers = get_n_outliers(series)
        insights.append(f"Number of outliers: {n_outliers}")
        insights.append(f"Ratio of outliers: {n_outliers/len(series)}")
        insights.append(f"Correlation with target: {corr[target][col]}")
        if col != target: insights.append(f"Mutual Information with target: {mi[col]}")
    

    return '\n'.join(insights)









# # df = pd.read_csv("/home/mark/Documents/Even_Newer_Python_Stuff/MPR-MAJOR/webserver/uploads/1/dataset.csv")
# # df = pd.read_csv('/home/mark/Documents/breast-cancer-wisconsin.csv')
# df = pd.read_csv('cancer-data.csv')
# pre_df = df.copy()

# insights = generate_insights(df, pre_df, 'Class')

# with open('insights.txt', 'w') as f:
#     f.write(insights)
    

# print('Insights have been generated and saved to insights.txt')
# chat = gemini_model.start_chat(history=[
#     {
#         "role": "user",
#         "parts": [PROMPT.format(insights=insights)]
#     },
# ])



# print('Chatbot is ready to answer questions')
# while True:
#     inp = input('> ')
#     response = chat.send_message(inp)
#     # inspect(response)
#     print(response.text)









    

