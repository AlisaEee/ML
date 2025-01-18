
# Деревья решений и метод ближайших соседей в задаче прогнозирования оттока клиентов телеком-оператора
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('./archive/telecom_churn.csv')
# Churn - отток клиентов (target-значение)
df['International plan'] = pd.factorize(df['International plan'])[0] # NO/YES values->int
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
df['Churn'] = df['Churn'].astype('int')
y = df['Churn']

# Можно удалить признак State он не сильно влияет на точность. Понижает
'''
labelencoder = LabelEncoder()
data_new = labelencoder.fit_transform(df['State'])
df['State'] = data_new
'''
df.drop(['State'], axis=1, inplace=True)

df.drop(['Churn'], axis=1, inplace=True)
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3,
random_state=17)

tree = DecisionTreeClassifier(max_depth=5, random_state=17)
# Find best params ПЕРЕБОР ПАРАМЕТРОВ
tree_params = {'max_depth': range(1,11),
'max_features': range(4,19)}
tree_grid = GridSearchCV(tree, tree_params,
cv=5, n_jobs=-1,
verbose=True)

tree_grid.fit(X_train, y_train)
tree_pred = tree_grid.predict(X_holdout)
print("Accuracy: ",accuracy_score(y_holdout, tree_pred)) # 0.925 - без tree_grid, c tree_grid - 0.94375
# KNN algoritm
knn = KNeighborsClassifier(n_neighbors=10)
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
knn_params = {'knn__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_params,
cv=5, n_jobs=-1,
verbose=True)
knn_grid.fit(X_train, y_train)
knn_pred = knn_grid.predict(X_holdout)
print("Accuracy knn: ",accuracy_score(y_holdout, knn_pred)) # 0.88 - без,