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

# Apply Decision Tree
from sklearn import tree

# clf = tree.DecisionTreeClassifier(max_depth=None, # You can set a maximum depth for the tree to prevent overfitting
#                                   min_samples_split=6, # The minimum number of samples required to split an internal node.
#                                   min_samples_leaf=4) # A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right.


# Apply random forest
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=None, # You can set a maximum depth for the tree to prevent overfitting
#                                           min_samples_split=7, # The minimum number of samples required to split an internal node.
#                                           min_samples_leaf=2, #A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right.
#                                           n_estimators=100)  # The number of trees in the forest
#
# clf = clf.fit(x_train, y_train)

# Produce visualizations of the tree graph.
# class_names = list(map(str, clf.classes_))
# plt.figure(figsize=(16, 8))
# tree.plot_tree(
#     decision_tree=clf,
#     max_depth=3,
#     feature_names=feature_names,
#     class_names=class_names,
#     filled=True
# )
# plt.show()

from sklearn.model_selection import cross_val_score

minSplit = [5, 10, 15, 20]
minLeaf = [3, 7, 11, 15]
minSplit5 = [5]
minSplit10 = [10]
minSplit15 = [15]
minSplit20 = [20]

plt.figure(figsize=(10,6))
plt.title('Accuracy vs. Min_Samples_Leaf (Decision Tree)')
plt.xlabel('Min_Samples_Leaf')
plt.ylabel('Accuracy')

#Loop to get accuracy data and change graphs

#Uncomment accuracy loop to print data
#Uncomment either decision tree or random forest to make graph

for sp in minSplit:
    cross_validation_accuracies = []
    for lf in minLeaf:
        print("Min Split - ", sp, " | Min Leaf - ", lf)
        clf = tree.DecisionTreeClassifier(max_depth=None,
                                          min_samples_split=sp,
                                          min_samples_leaf=lf)

        # clf = RandomForestClassifier(max_depth=None,
        #                              min_samples_split=sp,
        #                              min_samples_leaf=lf,
        #                              n_estimators=100)

        scores = cross_val_score(clf, df_feat, df['target'], cv=10)
        score = scores.mean()
        # print('10-fold cross-validation accuracy is:', score)
        cross_validation_accuracies.append(score)

        # Find the confusion matrix and accuracy metrics using this value for the hyperparameter and the training and test data that you created before.
        # print('confusion matrix and accuracy metrics for 30% test data')
        # clf.fit(x_train, y_train)
        # predictions_test = clf.predict(x_test)

        # Display confusion matrix
        # confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=clf.classes_)
        # confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
        #                                                           display_labels=clf.classes_)
        # confusion_matrix_display.plot()
        # plt.show()

        # Report Overall Accuracy, precision, recall, F1-score
        # class_names = list(map(str, clf.classes_))
        # print(metrics.classification_report(
        #     y_true=y_test,
        #     y_pred=predictions_test,
        #     target_names=list(map(str, class_names)),
        #     zero_division=0
        # ))
    newLabel = 'Minimum Sample Split: ', sp
    plt.plot(minLeaf, cross_validation_accuracies, label=newLabel, linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.legend()
plt.show()

