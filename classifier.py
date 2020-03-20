import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

df = pd.read_csv('mesothelioma_data.csv')

X = df.drop('duration of asbestos exposure', axis=1).values
y = df['duration of asbestos exposure'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn.fit(X_train, y_train)

print("Training accuracy: {}".format(knn.score(X_train, y_train)))
print("Testing accuracy: {}".format(accuracy_score(y_test, y_pred)))

figure = px.scatter(df, x='age', y='duration of asbestos exposure', color='class of diagnosis')
figure.show()
