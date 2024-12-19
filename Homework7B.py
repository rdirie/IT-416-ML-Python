import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
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

# Apply Support Vector Machines
# from sklearn.svm import SVC

# clf = clf = LogisticRegression(C=2) # Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
#
# clf = clf.fit(x_train, y_train)
#

# Predictions
# predictions_test = clf.predict(x_test)

# Display confusion matrix
# confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
# confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=clf.classes_)
# confusion_matrix_display.plot()
# from matplotlib import pyplot as plt
# plt.show()

# Report Overall Accuracy, precision, recall, F1-score
# class_names = list(map(str, clf.classes_))
# print(metrics.classification_report(
#     y_true=y_test,
#     y_pred=predictions_test,
#     target_names=class_names,
#     zero_division=0
# ))

# Hyperparameter Optimization
# Measuring the cross-validation accuracy for different values of hyperparameter and choosing the hyperparameter that results in the highest accuracy

# Apply SVM
# from sklearn.svm import SVC
#
# clf = SVC(C=1.0, # The smoothing parameter. Smaller values specify stronger regularization. If you have a lot of noisy observations you should decrease it.
#                       kernel='rbf', # 'rbf', 'linear', 'poly', 'sigmoid'
#                       degree=3) # Degree of the polynomial kernel function ('poly').

# Measuring the cross-validation accuracy for different values of hyperparameter and choosing the hyperparameter that results in the highest accuracy
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
cross_validation_accuracies = []
smooth = [0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0]
kernel = ['rbf', 'rbf', 'rbf', 'rbf', 'linear', 'linear', 'linear', 'linear', 'poly', 'poly', 'poly', 'poly', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
degrees = [2, 3, 4]
for c, k in zip(smooth, kernel):
    if k == 'poly':
        for d in degrees:
            clf = SVC(C=c, kernel=k, degree=d)
            scores = cross_val_score(clf, df_feat, df['target'], cv=10)
            score = scores.mean()
            print('10-fold cross-validation accuracy is:', score)
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
            print("C is ", c, " | Kernel ,", k, " | Degree ,", d)
            class_names = list(map(str, clf.classes_))
            print(metrics.classification_report(
                y_true=y_test,
                y_pred=predictions_test,
                target_names=list(map(str, class_names)),
                zero_division=0
            ))
    else:
        clf = SVC(C=c, kernel=k)
        scores = cross_val_score(clf, df_feat, df['target'], cv=10)
        score = scores.mean()
        print('10-fold cross-validation accuracy is:', score)
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
        print("C is ", c, " | Kernel ,", k)
        class_names = list(map(str, clf.classes_))
        print(metrics.classification_report(
            y_true=y_test,
            y_pred=predictions_test,
            target_names=list(map(str, class_names)),
            zero_division=0
        ))

plt.figure(figsize=(10, 6))
plt.plot(('0.5', '1.0', '1.5', '2.0'), cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. C-Values')
plt.xlabel('C-Value')
plt.ylabel('Accuracy')
plt.show()
