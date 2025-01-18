
# Деревья решений и метод ближайших соседей в задаче прогнозирования оттока клиентов телеком-оператора
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE 
def evaluateError(model, X_train, y_train, X_test, y_test, model_name):
    print("Model:",model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Mean ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    

df = pd.read_csv('./archive/credit_risk_dataset.csv')
# Предскажем статус кредита loan_status: 0,1
# Удаляем пропуски
df = df.dropna()
df.drop(['loan_int_rate'], axis=1)

# Преобразование категориальных признаков 
categorial = ['person_home_ownership','loan_intent','loan_grade', 'cb_person_default_on_file']
for feature in categorial:
    labelencoder = LabelEncoder()
    data_new = labelencoder.fit_transform(df[feature])
    df[feature] = data_new

# Split features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=17)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Address class imbalance using SMOTE
smote = SMOTE(random_state=17)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Logistic Regression
lr_model = LogisticRegression(random_state=17, class_weight='balanced')
evaluateError(lr_model,X_train_smote, y_train_smote, X_test_scaled, y_test, "Logistic Regression")

'''
# Hyperparameter Tuning
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

print("Best parameters:", grid_search.best_params_)
print("Best ROC AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_xgb_model = grid_search.best_estimator_
evaluate_model(best_xgb_model, X_train_smote, y_train_smote, X_test_scaled, y_test, "Best XGBoost Model"

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Print the top features
print("Top 15 features:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Most Important Features (XGBoost)')
plt.show()
'''


# Random Forest
rf_model = RandomForestClassifier(random_state=17, class_weight='balanced')
evaluateError(rf_model, X_train_smote, y_train_smote, X_test_scaled, y_test, "Random Forest")
