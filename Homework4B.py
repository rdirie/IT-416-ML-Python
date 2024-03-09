import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Read the Dataset

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


# Apply kNN
from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier(n_neighbors=9) # Default=5. Number of neighbors to use
# clf = clf.fit(x_train, y_train)

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

# Optimize k
from sklearn.model_selection import cross_val_score
cross_validation_accuracies = []
k_values = range(1, 15, 2)
for i in k_values:
    print('k is:', i)
    clf = KNeighborsClassifier(n_neighbors=i)
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
    confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=clf.classes_)
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

# Create a graph that shows the overall accuracy for different values of the hyperparameter.
plt.figure(figsize=(10,6))
plt.plot(k_values, cross_validation_accuracies, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K_value')
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.show()
