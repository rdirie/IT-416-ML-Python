import pandas as pd
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

# Scatter plot of training samples and SVM classifier in a 2d space.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
# Reduce the number of features to 2, so you can plot them
pca = PCA(n_components=2) # Create a PCA Object that will generate two features from the existing features
pca = pca.fit(x_train) # Fit PCA to training data
x_train_2 = pca.transform(x_train)
x_test_2 = pca.transform(x_test)
# Apply Machine Learning Models
clf_pca = SVC(C=1.0, kernel='linear', degree=3)
# clf_pca = SVC(C=1.0, kernel='poly', degree=3)
# clf_pca = SVC(C=1.0, kernel='sigmoid', degree=3)
# clf_pca = SVC(C=1.0, kernel='rbf', degree=3)
# clf_pca = clf = tree.DecisionTreeClassifier(max_depth=None,
#                                    min_samples_split=6,
#                                   min_samples_leaf=4)
# clf_pca = RandomForestClassifier(max_depth=None,
#                                           min_samples_split=7,
#                                           min_samples_leaf=2,
#                                           n_estimators=100)
# clf_pca = LogisticRegression(C=2)
# clf_pca = GaussianNB(priors=None)
# clf_pca = KNeighborsClassifier(n_neighbors=9)
# clf_pca = linear_model.LinearRegression()
# clf_pca = linear_model.Ridge(alpha=.5,
#                  random_state=0,
#                  )
# clf_pca = linear_model.Lasso(alpha=0.1,
#                      random_state=0,
#                      )


clf_pca = clf_pca.fit(x_train_2, y_train)
predictions_test = clf_pca.predict(x_test_2)
# Scatter plot
y_train = y_train.tolist()
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01), np.arange(second_dimension_min, second_dimension_max, .01))
Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Draw contour line
plt.contour(xx, yy, Z)
plt.title('SVM classifier with (machine learning model)')
plt.axis('off')
plt.show()
