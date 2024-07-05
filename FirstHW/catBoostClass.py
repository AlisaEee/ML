from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
import pandas as pd
from scipy.sparse import hstack,csr_matrix
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score,balanced_accuracy_score
base = 'C:/Users/eaa-h/ML/FirstHW/Lab_EnteringML/'
data = pd.read_csv(base + 'organisations.csv')
clean_data1 = data[data['average_bill'].notnull()]
clean_data = clean_data1[clean_data1['average_bill'] <= 2500]
clean_data_train, clean_data_test = train_test_split(clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)
clean_data_train['modified_features'] = clean_data_train['rubrics_id'] + ' q ' + clean_data_train['features_id']

clean_data_test['modified_features'] = clean_data_test['rubrics_id'] + ' q ' + clean_data_test['features_id']
common_features = set(clean_data_train['modified_features']) & set(clean_data_test['modified_features'])
clean_data_test['modified_features'] = clean_data_test['modified_features'].replace(list(common_features), 'other')

# Создание dummy-переменных для столбца city
city_dummies = pd.get_dummies(clean_data_train['city'])
# Создание разреженной матрицы для столбца rating
rating_sparse = csr_matrix(clean_data_train['rating'].values.reshape(-1,1))
# Преобразование категориальных признаков в отдельные переменные
rubric_dummies = pd.get_dummies(clean_data_train['rubrics_id'])
feature_dummies = pd.get_dummies(clean_data_train['features_id'])
# Соединение всех признаков в одну разреженную матрицу
#sparse_data_train = csr_matrix(pd.concat([city_dummies,rating_sparse,rubric_dummies, feature_dummies], axis=1))
sparse_data_train = hstack([city_dummies, rubric_dummies,feature_dummies, rating_sparse])

# Создание dummy-переменных для столбца city
city_dummies1 = pd.get_dummies(clean_data_test['city'])
# Создание разреженной матрицы для столбца rating
rating_sparse1 = csr_matrix(clean_data_test['rating'].values.reshape(-1,1))
# Преобразование категориальных признаков в отдельные переменные
rubric_dummies1 = pd.get_dummies(clean_data_test['rubrics_id'])
feature_dummies1 = pd.get_dummies(clean_data_test['features_id'])
sparse_data_test = hstack([city_dummies1, rubric_dummies1,feature_dummies1, rating_sparse1])

clf = CatBoostClassifier()
clf.fit(sparse_data_train, clean_data_train['average_bill'])
predictions = clf.predict(sparse_data_train)
print(predictions)
# Оценка производительности модели
mse = mean_squared_error(clean_data_train['average_bill'], predictions)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")
print("ACCURACY:",balanced_accuracy_score(clean_data_train['average_bill'], predictions))

#TEST DATA
clf = CatBoostClassifier()
clf.fit(sparse_data_test, clean_data_test['average_bill'])
predictions = clf.predict(sparse_data_test)
print(predictions)
# Оценка производительности модели
mse = mean_squared_error(clean_data_test['average_bill'], predictions)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")
print("ACCURACY:",balanced_accuracy_score(clean_data_test['average_bill'], predictions))