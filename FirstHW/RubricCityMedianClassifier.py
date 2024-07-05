from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score,balanced_accuracy_score
base = 'C:/Users/eaa-h/ML/FirstHW/Lab_EnteringML/'
data = pd.read_csv(base + 'organisations.csv')
clean_data1 = data[data['average_bill'].notnull()]
clean_data = clean_data1[clean_data1['average_bill'] <= 2500]
clean_data_train, clean_data_test = train_test_split(clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)
cntTrain = Counter(data['rubrics_id']) 
dictionaryTrain = dict(cntTrain)
data = ['other' if dictionaryTrain[x] < 100 else x for x in clean_data_train['rubrics_id']]
clean_data_train['modified_rubrics'] = data
class RubricCityMedianClassifier(ClassifierMixin):
    def __init__(self):
        self.city_median_dict = None

    def fit(self, X, y):
        data = pd.DataFrame({'modified_rubrics': X['modified_rubrics'], 'city': X['city'], 'average_bill': y})
        self.city_median_dict = data.groupby(['modified_rubrics', 'city'])['average_bill'].median().to_dict()
        return self

    def predict(self, X):
        predictions = []
        for ind, row in X.iterrows():
            key = (row['modified_rubrics'], row['city'])
            if key in self.city_median_dict:
                predictions.append(self.city_median_dict[key])
            else:
                # Если для данной комбинации нет данных, можно вернуть общую медиану или другое значение по умолчанию
                predictions.append(np.median(list(self.city_median_dict.values())))
        return np.array(predictions)

reg = RubricCityMedianClassifier()
reg.fit(X=clean_data_train,y=clean_data_train['average_bill'])
predictions = reg.predict(X=clean_data_train)
print(predictions)
# Оценка производительности модели
mse = mean_squared_error(clean_data_train['average_bill'], predictions)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")
print("ACCURACY:",balanced_accuracy_score(clean_data_train['average_bill'], predictions))