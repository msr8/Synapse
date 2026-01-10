# bayes_opt_svc_plot.py

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set up BayesSearchCV
opt = BayesSearchCV(
    estimator=SVC(),
    search_spaces={
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'kernel': ['rbf', 'poly']
    },
    n_iter=3,
    random_state=42,
    cv=3,
    n_jobs=-1
)

# Fit the model
opt.fit(X_train, y_train)

# Access optimizer results (first space)
print(opt.optimizer_results_)
result = opt.optimizer_results_[0]

# Print best results
print("Best score:", opt.best_score_)
print("Best parameters:", opt.best_params_)

# Plot results
plot_convergence(result)
plot_objective(result)
plot_evaluations(result)
plt.show()
