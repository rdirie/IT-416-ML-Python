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

# Apply Naive Bayes
from sklearn.naive_bayes import GaussianNB

# clf = GaussianNB(priors=None) # An array whose size is equal to the number of classes. If not specified or None, the priors are adjusted based on relative class frequencies. Set it to [0.5 0.5] if you want equal priors.
#
# clf = clf.fit(x_train, y_train)
#
# print('Class priors are: ', clf.class_prior_)

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
from sklearn.model_selection import cross_val_score
cross_validation_accuracies = []
priors = ([0, 1], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1, 0])
for p in priors:
    print('Priors are:', p)
    clf = GaussianNB(priors=p)
    scores = cross_val_score(clf, df_feat, df['target'], cv=10)
    score = scores.mean()
    print('10-fold cross-validation accuracy is:', score)
    cross_validation_accuracies.append(score)

    # Find the confusion matrix and accuracy metrics using this value for the hyperparameter and the training and test data that you created before.
    print('confusion matrix and accuracy metrics for 30% test data')
    clf.fit(x_train, y_train)
    predictions_test = clf.predict(x_test)

    # Display confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
    confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                              display_labels=clf.classes_)
    confusion_matrix_display.plot()
    plt.show()

    # Report Overall Accuracy, precision, recall, F1-score
    class_names = list(map(str, clf.classes_))
    print(metrics.classification_report(
        y_true=y_test,
        y_pred=predictions_test,
        target_names=list(map(str, class_names)),
        zero_division=0
    ))

plt.figure(figsize=(10, 6))
plt.plot(('0-100', '10-90', '20-80', '30-70', '40-60', '50-50', '60-40', '70-30', '80-20', '90-10', '100-0'), cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. Prior')
plt.xlabel('Prior')
plt.ylabel('Accuracy')
plt.show()

