import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm 
import optuna
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.model_selection import train_test_split
import joblib

train = pd.read_csv(r'C:\Users\Cars24\Downloads\Classification with an Academic Success\data\train.csv')
test = pd.read_csv(r'C:\Users\Cars24\Downloads\Classification with an Academic Success\data\test.csv')

# remove id columns from train and test data 

train=train.drop('id',axis = 1)
test = test.drop('id',axis =1)

X = train.drop('Target',axis = 1)
y = train['Target']

selector = SelectKBest(score_func=f_classif,k=12)
reduced_data = selector.fit_transform(X,y)
selected_feature = selector.get_support()
selected_columns = X.columns[selected_feature]

class column_selector(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        self.cols = cols 
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.cols]

params = {'n_estimators': 255,
 'learning_rate': 0.03799816394205466,
 'max_depth': 5,
 'min_samples_split': 12,
 'min_samples_leaf': 14,
 'subsample': 0.9164457669640189}

pipeline = Pipeline(steps=[
    ('selector',column_selector(cols = selected_columns)),
    ('scale' , StandardScaler()),
    ('gradient_boost' , GradientBoostingClassifier(**params))
])

le = LabelEncoder()
y_encode = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(X,y_encode,test_size=0.2,random_state=42)

pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)

acc = accuracy_score(y_test,y_pred)

print(acc)

joblib.dump(pipeline,"model.joblib")