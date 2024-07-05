from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score
base = 'C:/Users/eaa-h/ML/FirstHW/Lab_EnteringML/'
data = pd.read_csv(base + 'organisations.csv')
clean_data1 = data[data['average_bill'].notnull()]
clean_data = clean_data1[clean_data1['average_bill'] <= 2500]
clean_data_train, clean_data_test = train_test_split(clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)
class CityMeanRegressor(RegressorMixin):
    def __init__(self):
        self.city_means = None

    def fit(self, X, y):
        data = pd.DataFrame({'city': X['city'], 'average_bill': y})
        self.city_means = data.groupby('city')['average_bill'].mean().to_dict()
        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        predictions = [self.city_means[city] if city in self.city_means else 0 for city in X['city']]
        return np.array(predictions)

reg = CityMeanRegressor()
reg.fit(X=clean_data_train,y=clean_data_train['average_bill'])
predictions = reg.predict(X=clean_data_train)
print(predictions)
# Оценка производительности модели
mse = mean_squared_error(clean_data_train['average_bill'], predictions)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")