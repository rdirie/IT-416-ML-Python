import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
df = pd.read_csv('/Users/rayyandirie/Desktop/GMU - Spring 2024/IT 416/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split the training and testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Apply MLP
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
# clf = MLPClassifier(random_state=1, # Pass an int for reproducible results across multiple function calls.
#                     hidden_layer_sizes=(20, 20), # tuple, length = n_layers - 2, default=(100,). The ith element represents the number of neurons in the ith hidden layer.
#                     activation='relu', # {'identity', 'logistic', 'tanh', 'relu'}, default='relu'. Activation function for the hidden layer.
#                     solver='adam', # {'lbfgs', 'sgd', 'dam'}, default='adam'. The solver for weight optimization.
#                     alpha=0.00001, # L2 penalty (regularization term) parameter.
#                     batch_size='auto', # int, default='auto'. Size of minibatches for stochastic optimizers. When set to 'auto', batch_size=min(200, n_samples).
#                     learning_rate='adaptive', # {'constant', 'invscaling', 'adaptive'}, default='constant'. Learning rate schedule for weight updates.
#                     # 'constant' is a constant learning rate given by 'learning_rate_init'.
#                     # 'invscaling' gradually decreases the learning rate at each time step 't' using an inverse scaling exponent of 'power_t'. effective_learning_rate = learning_rate_init / pow(t, power_t)
#                     # 'adaptive' keeps the learning rate constant to 'learning_rate_init' as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if 'early_stopping' is on, the current learning rate is divided by 5.
#                     learning_rate_init=0.001, # The initial learning rate used. It controls the step-size in updating the weights.
#                     max_iter=1000, # Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.
#                     shuffle=True, # Whether to shuffle samples in each iteration.
#                     tol=0.0001, # default=1e-4. Tolerance for the optimization.
#                     # When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, convergence is considered to be reached and training stops.
#                     early_stopping=False, # Whether to use early stopping to terminate training when validation score is not improving.
#                     # If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
#                     n_iter_no_change=10) # Maximum number of epochs to not meet tol improvement. Only effective when solver='sgd' or 'adam'.
#
# clf = clf.fit(x_train, y_train)
#
#
# print(clf.n_iter_) # int. The number of iterations the solver has run.
#
# print(clf.loss_curve_) # List of shape (n_iter_) The ith element in the list represents the loss at the ith iteration.

plt.figure(figsize=(10,6))
plt.title('Accuracy vs. Hidden Layer Sizes')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')

#Loop to get accuracy data and change graphs

activationFunction = ['relu', 'identity', 'logistic', 'tanh']
hiddenLayerSize = [(10,), (20,), (10,10), (20,20)]

for af in activationFunction:
    cross_validation_accuracies = []
    for hl in hiddenLayerSize:
        print("Activation Function - ", af, " | Hidden Layer Size - ", hl)

        clf = MLPClassifier(random_state=1, # Pass an int for reproducible results across multiple function calls.
                            hidden_layer_sizes=hl, # tuple, length = n_layers - 2, default=(100,). The ith element represents the number of neurons in the ith hidden layer.
                            activation=af, # {'identity', 'logistic', 'tanh', 'relu'}, default='relu'. Activation function for the hidden layer.
                            solver='adam', # {'lbfgs', 'sgd', 'dam'}, default='adam'. The solver for weight optimization.
                            alpha=0.00001, # L2 penalty (regularization term) parameter.
                            batch_size='auto', # int, default='auto'. Size of minibatches for stochastic optimizers. When set to 'auto', batch_size=min(200, n_samples).
                            learning_rate='adaptive', # {'constant', 'invscaling', 'adaptive'}, default='constant'. Learning rate schedule for weight updates.
                            # 'constant' is a constant learning rate given by 'learning_rate_init'.
                            # 'invscaling' gradually decreases the learning rate at each time step 't' using an inverse scaling exponent of 'power_t'. effective_learning_rate = learning_rate_init / pow(t, power_t)
                            # 'adaptive' keeps the learning rate constant to 'learning_rate_init' as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if 'early_stopping' is on, the current learning rate is divided by 5.
                            learning_rate_init=0.001, # The initial learning rate used. It controls the step-size in updating the weights.
                            max_iter=1000, # Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.
                            shuffle=True, # Whether to shuffle samples in each iteration.
                            tol=0.0001, # default=1e-4. Tolerance for the optimization.
                            # When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, convergence is considered to be reached and training stops.
                            early_stopping=False, # Whether to use early stopping to terminate training when validation score is not improving.
                            # If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
                            n_iter_no_change=10) # Maximum number of epochs to not meet tol improvement. Only effective when solver='sgd' or 'adam'.

        scores = cross_val_score(clf, df_feat, df['target'], cv=10)
        score = scores.mean()
        # print('10-fold cross-validation accuracy is:', score)
        cross_validation_accuracies.append(score)

        # Find the confusion matrix and accuracy metrics using this value for the hyperparameter and the training and test data that you created before.
        print('confusion matrix and accuracy metrics for 30% test data')
        clf.fit(x_train, y_train)
        predictions_test = clf.predict(x_test)

        # Display confusion matrix
        # confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
        # confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
        #                                                           display_labels=clf.classes_)
        # confusion_matrix_display.plot()
        # plt.show()

        # Report Overall Accuracy, precision, recall, F1-score
        class_names = list(map(str, clf.classes_))
        print(metrics.classification_report(
            y_true=y_test,
            y_pred=predictions_test,
            target_names=list(map(str, class_names)),
            zero_division=0
        ))
    newLabel = 'Activation Function - ', af
    plt.plot(['(10,)', '(20,)', '(10,10)', '(20,20)'], cross_validation_accuracies, label=newLabel, linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)

plt.legend()
plt.show()