from sklearn.base import ClassifierMixin
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,balanced_accuracy_score
base = 'C:/Users/eaa-h/ML/FirstHW/Lab_EnteringML/'
data = pd.read_csv(base + 'organisations.csv')
clean_data1 = data[data['average_bill'].notnull()]
clean_data = clean_data1[clean_data1['average_bill'] <= 2500]
print(clean_data['average_bill'])
clean_data_train, clean_data_test = train_test_split(clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)

class MostFrequentClassifier(ClassifierMixin):
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Training data features
        y : array like, shape = (_samples,)
            Training data targets
        '''
        self.mode_ = mode(y)[0]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Data to predict
        '''
        return np.full(shape=X.shape[0], fill_value=self.mode_)
reg = MostFrequentClassifier()
reg.fit(X=clean_data_train,y=clean_data_train['average_bill'])
predictions = reg.predict(X=clean_data_train)
print(predictions)
# Оценка производительности модели
mse = mean_squared_error(clean_data_train['average_bill'], predictions)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print(f"Root Mean Square Error (RMSE): {rmse}")
print("ACCURACY:",balanced_accuracy_score(clean_data_train['average_bill'], predictions))