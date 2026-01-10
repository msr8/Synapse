<div align="center">

<!-- https://coolors.co/gradient-palette/f72585-066da5?number=3 -->

[![GitHub stars](https://img.shields.io/github/stars/msr8/synapse?color=F72585&labelColor=302D41&style=for-the-badge)](https://github.com/msr8/synapse/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/msr8/synapse?color=7F4995&labelColor=302D41&style=for-the-badge)](https://github.com/msr8/synapse/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/msr8/synapse?color=066DA5&labelColor=302D41&style=for-the-badge)](https://github.com/msr8/synapse/issues)

</div>



<br><br>



# Abstract

**Problem Statement:** Extracting meaningful insights from raw datasets and selecting the most effective machine learning models remains a significant challenge for both non-technical and technical users. Non-technical users and analysts often face steep learning curves due to complex tools and the need for coding expertise. On the other hand, technical users struggle with fragmented workflows that lack intuitive interfaces for rapid experimentation, hyper-parameter tuning, and performance comparison. This disconnect hinders efficient model development and slows down decision-making across teams

**Context and Background:** In the current data-driven era, organizations and individuals increasingly rely on data analysis for strategic actions. However, most available tools require programming knowledge or familiarity with data science workflows. This creates a barrier for non-technical users and business professionals who need to make sense of data without specialized skills. Even technical users encounter inefficiencies due to disjointed tools and unintuitive interfaces, making it harder to iterate quickly, fine-tune models, and compare results effectively

**Purpose and Contribution:** Synapse aims to democratize data analysis by developing a no-code, web-based platform that enables users to upload datasets, perform exploratory data analysis (EDA), and select the most appropriate machine learning model through a simple, conversational interface. The system bridges the gap between usability and advanced analytics by combining automation with natural language interaction

**Methods and Approach:** Synapse includes a user-friendly web interface with two modes: a visual dashboard for EDA and bayesian optimization, and a chatbot for natural language queries. Upon uploading a dataset file, the system automatically handles data preprocessing such as cleaning, encoding, and scaling. Users can visualize the dataset and interact with the chatbot. For model selection, Bayesian optimization is used to identify the best-fit algorithm for classification

**Results and Conclusion:** Synapse successfully simplifies complex data tasks, enabling users to analyze and interpret their datasets without writing code. It demonstrates that combining automation, natural language processing, and model optimization can make machine learning more accessible, thereby enhancing decision-making for users across technical and non-technical domains

<br>

### Features

1) Engineered a **full-stack real-time ML platform** enabling seamless dataset upload, processing, and model hyperparameter finetuning
2) Integrated **Bayesian Optimization** to autonomously tune hyperparameters for diverse models like **LightGBM, SVM, and Random Forest**
3) Implemented an **automated EDA pipeline** generating insightful and interactive visualizations including correlation matrices, mutual information heatmaps, and pairplots
4) Developed a **robust customisable preprocessing engine** with intelligent missing value handling, encoding, feature selection, and scaling
5) Embedded a **Generative AI chatbot** to provide interactive, data-driven insights and statistical interpretations to both, technical and non-technical users

<!-- 
Based on the provided codebase, here are 5 points summarizing the project in your requested style:

1) Engineered a **real-time ML pipeline** using **Flask** and **Socket.IO**, enabling seamless dataset upload, processing, and model training.
2) Implemented automated **Exploratory Data Analysis** utilizing **Matplotlib and Pygal** to generate insightful visualizations like correlation heatmaps and pairplots.
3) Developed a robust **preprocessing engine** featuring advanced **feature selection algorithms** (Mutual Info, Chi2, ANOVA) and automated scaling.
4) Integrated **Bayesian Optimization** to autonomously tune hyperparameters for diverse models like **LightGBM, SVM, and Random Forest**.
5) Embedded a **Generative AI chatbot** leveraging the **Gemini API** to provide interactive, data-driven insights and statistical interpretations to users.

1) Engineered a **full-stack machine learning platform** using Flask and Socket.IO to facilitate real-time data processing and asynchronous model training feedback
2) Integrated **Google's Gemini API** to power an intelligent chatbot that interprets datasets and generates natural language statistical insights for users 
3) Implemented **automated Bayesian hyperparameter optimization** using Scikit-Optimize to efficiently tune and evaluate diverse classification models like LightGBM, SVM, and Random Forest 
4) Developed an **automated Exploratory Data Analysis (EDA)** suite leveraging Matplotlib and Seaborn to dynamically generate correlation heatmaps, pairplots, and distribution charts 
5) Built a **comprehensive data preprocessing pipeline** featuring advanced feature selection methods (Mutual Information, ANOVA, Chi-Square) and robust handling of missing values and scaling

1) Implemented **Bayesian hyperparameter optimization** across 10 classification algorithms with real-time progress updates via WebSockets
2) Integrated **Google Gemini AI** to provide an interactive chatbot that answers questions about datasets using EDA and model optimization insights
3) Developed an **automated EDA pipeline** generating interactive visualizations including correlation matrices, mutual information heatmaps, and pairplots
4) Engineered a **customizable preprocessing pipeline** with intelligent missing value handling, multiple feature selection methods, and four scaling techniques
5) Utilized **Flask-SocketIO** for real-time bidirectional communication, enabling live updates during model training and data processing operations
-->

<br>

### Tech Stack

1) **Backend & Frameworks**
   - **Python (Flask):** The core web framework used to build the application
   - **Flask-SocketIO:** Enables real-time, bi-directional communication for the EDA and training logs
   - **Flask-SQLAlchemy:** ORM for database management
   - **Flask-Dance:** Handles Google OAuth 2.0 authentication
2) **Machine Learning & AI**
   - **Scikit-Learn:** Used for standard algorithms (SVM, KNN, Random Forest, etc.) and metrics
   - **Scikit-Optimize (skopt):** Powers the Bayesian Optimization engine for hyperparameter tuning
   - **XGBoost & LightGBM:** Advanced gradient boosting frameworks integrated into the pipeline
3) **Visualization**
   - **Matplotlib & Seaborn:** Generates static charts like correlation heatmaps and pairplots (rendered to Base64)
   - **Pygal:** Used for interactive vector-based (SVG) visualizations
4) **Frontend**
   - **HTML5 / CSS3 / JavaScript:** Core technologies for the user interface
   - **GSAP (GreenSock):** Used for advanced animations and scroll triggers
   - **Motion (Motion One):** A modern animation library for UI transitions
   - **JSZip & FileSaver.js:** Allows users to zip and download generated charts directly from the browser
5) **Database**
   - **SQLite:** A fast and simple database used for storing user data and task information



<br><br>



# Usage

To run the application locally, follow these steps:

First of all, ensure that you have [git](https://git-scm.com/) and [Python 3.8+](https://www.python.org/downloads/) installed on your machine. Then, run the following commands:
```bash
# Clone the repository
git clone https://github.com/msr8/synapse
cd synapse/src
# Install the required dependencies
pip install -r requirements.txt
# Run the flask application
python app.py
```

The application will be accessible at `http://127.0.0.1:5000` in your web browser

> [!WARNING]
> These instructions are intended for local development only. For production deployment, use a production-ready server like [Gunicorn](https://gunicorn.org/) or [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/), and consider using a reverse proxy like [Nginx](https://www.nginx.com/)



<br><br>



# Endpoints

| URL Path                               | Description                                      |
|----------------------------------------|--------------------------------------------------|
| `/`                                    | Landing page                                     |
| `/learn-more`                          | Information page about the project               |
| `/dashboard`                           | User dashboard displaying tasks                  |
| `/login`                               | User login page                                  |
| `/signup`                              | User registration page                           |
| `/logout`                              | Logs the user out                                |
| `/login/google-authorised/`            | Google OAuth callback URL                        |
| `/task/<int:task_id>`                  | Main interface for a specific task               |
| `/api/auth/login`                      | API to handle user login                         |
| `/api/auth/signup`                     | API to handle user registration                  |
| `/api/auth/change-username`            | API to update the current user's username        |
| `/api/auth/change-password`            | API to update the current user's password        |
| `/api/upload`                          | API to handle dataset uploads                    |
| `/api/task/set-target`                 | API to set the target column for a task          |
| `/api/task/change-taskname`            | API to rename a specific task                    |
| `/api/task/delete-task`                | API to delete a task                             |
| `/api/task/chatbot/initialise`         | API to start the LLM chat session                |
| `/api/task/chatbot/chat`               | API to send a message to the chatbot             |
| `/api/task/chatbot/reset`              | API to clear chat history                        |



<br><br>



# Search Space

We optimise over the following classification models using Bayesian optimization to find the best model and hyperparameters for a given dataset:

<details>
<summary>1) K-Nearest-Neighbours</summary>

| Hyperparameter | Description                          | Type          | Range / Values                                                             |
|----------------|--------------------------------------|---------------|----------------------------------------------------------------------------|
| `n_neighbors`  | Number of neighbors to use          | Integer       | 1 to 30                                                                     |
| `weights`      | Weight function used in prediction  | Categorical   | `uniform`, `distance`                                                       |
| `metric`       | Distance metric to use              | Categorical   | `chebyshev`, `cosine`, `euclidean`, `manhattan`, `minkowski`, `sqeuclidean` |
</details>

<details>
<summary>2) Support Vector Machine</summary>

| Hyperparameter | Description                          | Type          | Range / Values                                     |
|----------------|--------------------------------------|---------------|----------------------------------------------------|
| `C`            | Regularization parameter             | Float         | 1e-4 to 1e+4 (log-uniform)                         |
| `kernel`       | Kernel type to be used               | Categorical   | `rbf`,`sigmoid`, `poly`                            |
| `degree`       | Degree of the polynomial kernel      | Integer       | 1 to 3                                             |
| `gamma`        | Kernel coefficient                   | Categorical   | `scale`                                            |
</details>

<details>
<summary>3) Logistic Regression</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `C` | Inverse of regularization strength | Float | 1e-6 to 1e+6 (log-uniform) |
| `penalty` | Norm used in penalization | Categorical | `l1`, `l2` |
| `solver` | Optimization algorithm | Categorical | `liblinear`, `saga` |
</details>

<details>
<summary>4) Decision Tree</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `criterion` | Function to measure split quality | Categorical | `gini`, `entropy` |
| `splitter` | Strategy used to choose split | Categorical | `best`, `random` |
| `max_depth` | Maximum depth of the tree | Integer | 1 to 10 |
| `min_samples_split` | Min samples required to split node | Integer | 2 to 10 |
| `min_samples_leaf` | Min samples required at leaf node | Integer | 1 to 10 |
| `max_features` | Number of features to consider | Categorical | `None`, `sqrt`, `log2` |
</details>

<details>
<summary>5) Random Forest</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `n_estimators` | Number of trees in the forest | Integer | 10 to 100 |
| `criterion` | Function to measure split quality | Categorical | `gini`, `entropy` |
| `max_depth` | Maximum depth of the tree | Integer | 1 to 10 |
| `min_samples_split` | Min samples required to split node | Integer | 2 to 10 |
| `min_samples_leaf` | Min samples required at leaf node | Integer | 1 to 10 |
| `max_features` | Number of features to consider | Categorical | `None`, `sqrt`, `log2` |
</details>

<details>
<summary>6) Extra Trees</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `n_estimators` | Number of trees in the forest | Integer | 10 to 100 |
| `criterion` | Function to measure split quality | Categorical | `gini`, `entropy` |
| `max_depth` | Maximum depth of the tree | Integer | 1 to 10 |
| `min_samples_split` | Min samples required to split node | Integer | 2 to 10 |
| `min_samples_leaf` | Min samples required at leaf node | Integer | 1 to 10 |
| `max_features` | Number of features to consider | Categorical | `None`, `sqrt`, `log2` |
</details>

<details>
<summary>7) Gradient Boosting</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `n_estimators` | Number of boosting stages | Integer | 10 to 100 |
| `learning_rate` | Shrinks contribution of each tree | Float | 1e-6 to 1 (log-uniform) |
| `max_depth` | Maximum depth of estimators | Integer | 1 to 10 |
| `min_samples_split` | Min samples required to split node | Integer | 2 to 10 |
| `min_samples_leaf` | Min samples required at leaf node | Integer | 1 to 10 |
| `max_features` | Number of features to consider | Categorical | `None`, `sqrt`, `log2` |
</details>

<details>
<summary>8) Light Gradient Boosting Machine (LGBM)</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `n_estimators` | Number of boosted trees | Integer | 10 to 100 |
| `learning_rate` | Boosting learning rate | Float | 1e-6 to 1 (log-uniform) |
| `max_depth` | Maximum tree depth | Integer | -1 to 15 |
| `num_leaves` | Max tree leaves for base learners | Integer | 10 to 50 |
| `min_child_samples` | Min data needed in a leaf | Integer | 5 to 20 |
| `subsample` | Subsample ratio of training instance | Float | 0.5 to 1.0 |
| `colsample_bytree` | Subsample ratio of columns per tree | Float | 0.5 to 1.0 |
| `reg_alpha` | L1 regularization term | Float | 0.0 to 5.0 |
| `reg_lambda` | L2 regularization term | Float | 0.0 to 5.0 |
</details>

<details>
<summary>9) Ada Boost</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `n_estimators` | Maximum number of estimators | Integer | 10 to 100 |
| `learning_rate` | Weight applied to each classifier | Float | 1e-6 to 1 (log-uniform) |
</details>

<details>
<summary>10) Bagging</summary>

| Hyperparameter | Description | Type | Range / Values |
| --- | --- | --- | --- |
| `n_estimators` | Number of base estimators | Integer | 10 to 100 |
| `max_samples` | Number of samples to draw | Float | 0.1 to 1.0 |
| `max_features` | Number of features to draw | Float | 0.1 to 1.0 |
| `bootstrap` | Draw samples with replacement | Boolean | `True`, `False` |
| `bootstrap_features` | Draw features with replacement | Boolean | `True`, `False` |
</details>



<br><br>



# Screenshots

<div align="center">

<!--  '1 home.png'          '11 faq.png'      '3 target selection.png'   '5 corr and mi.png'   '7 bayes.png'     '9 dashboard.png'
 '10 learn more.png'   '2 options.png'   '4 feature charts.png'     '6 pairplot.png'      '8 chatbot.png'   -->

![Landing Page](./screenshots/1%20home.png)
Figure 1: Home Page
<br>

![Configurable Options](./screenshots/2%20options.png)
Figure 2: Configurable Options for EDA, Preprocessing, and Bayesian Optimization
<br>

![Target Selection](./screenshots/3%20target%20selection.png)
Figure 3: Target Column Selection
<br>

![Feature Charts](./screenshots/4%20feature%20charts.png)
Figure 4: Feature Columns Distributions
<br>

![Correlation and Mutual Info](./screenshots/5%20corr%20and%20mi.png)
Figure 5: Correlation Heatmap and Mutual Information Heatmap
<br>

![Pairplot](./screenshots/6%20pairplot.png)
Figure 6: Pairplot Visualization
<br>

![Bayesian Optimization](./screenshots/7%20bayes.png)
Figure 7: Real-time Logs Bayesian Optimization Results
<br>

![Chatbot Interface](./screenshots/8%20chatbot.png)
Figure 8: Chatbot Interface
<br>

![Dashboard](./screenshots/9%20dashboard.png)
Figure 9: User Dashboard
<br>

![Learn More Page](./screenshots/10%20learn%20more.png)
Figure 10: Learn More Page
<br>

![FAQ Page](./screenshots/11%20faq.png)
Figure 11: FAQs


