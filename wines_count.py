from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import classification_report
import scipy.sparse
import time
import pickle
from joblib import dump, load

import pandas as pd
import numpy as np
import xgboost, textblob, string



data = pd.read_csv('winemag-data_first150k.csv', index_col=0)

conditions = [
    (data['points'] <= 84),
    (data['points'] > 84) & (data['points'] <= 88),
    (data['points'] > 88) & (data['points'] <= 92),
    (data['points'] > 92) & (data['points'] <= 96),
    (data['points'] > 96) & (data['points'] <= 100)
    ]

# create a list of the values we want to assign for each condition
class_list = [0,1,2,3,4]#['Low', 'OK', 'Good', 'Very Good', 'Excellent']

# create a new column and use np.select to assign values to it using our lists as arguments
data['class'] = np.select(conditions, class_list)

data = data[['description','winery', 'country','class']] # selecting columns
data = data.dropna() # deleting na rows

data["united"] = data["country"] + " " + data['winery'] + " " + data["description"]

X_train, X_test, y_train, y_test = model_selection.train_test_split(data['united'],
                                                                    data['class'],
                                                                    test_size=0.33,
                                                                    random_state=42)
                                                                    


encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(data['united'])

# transform the training and validation data using count vectorizer object
X_train =  count_vect.transform(X_train)
X_test =  count_vect.transform(X_test)



rf = load('count/RandomForestClassifier_count.joblib')
nb = load('count/MultinomialNB_count.joblib')
lg = load('count/LogisticRegression_count.joblib')
mlp = load('count/MLPClassifier_count.joblib')
dt = load('count/DecisionTreeClassifier_count.joblib')
xgb = load('count/XGBClassifier_count.joblib')

rf_pred = rf.predict(X_test)
nb_pred = nb.predict(X_test)
lg_pred = lg.predict(X_test)
mlp_pred = mlp.predict(X_test)
dt_pred = dt.predict(X_test)
xgb_pred = xgb.predict(X_test)


print( "rf",metrics.accuracy_score(rf_pred, y_test),classification_report(y_test,rf_pred), sep = '\n')
print( "nb",metrics.accuracy_score(nb_pred, y_test),classification_report(y_test,nb_pred),sep = '\n')
print( "lg",metrics.accuracy_score(lg_pred, y_test),classification_report(y_test,lg_pred),sep = '\n')
print( "dt",metrics.accuracy_score(dt_pred, y_test),classification_report(y_test,dt_pred),sep = '\n')
print( "mlp",metrics.accuracy_score(mlp_pred, y_test),classification_report(y_test,mlp_pred),sep = '\n')
print( "xgb",metrics.accuracy_score(xgb_pred, y_test),classification_report(y_test,xgb_pred),sep = '\n')
