# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
data_df = pd.read_csv('DataSet - Daibates.csv')

#Imputing NULL Values
data_df['Number of times Pregnant'].fillna(data_df['Number of times Pregnant'].mean(), inplace = True)
data_df['Plasma Concentration'].fillna(data_df['Plasma Concentration'].median(), inplace = True)
data_df['Diastolic BP'].fillna(data_df['Diastolic BP'].mean(), inplace = True)
data_df['Triceps Skin fold thickness'].fillna(data_df['Triceps Skin fold thickness'].mean(), inplace = True)
data_df['insulin'].fillna(data_df['insulin'].median(), inplace = True)
data_df['BMI'].fillna(data_df['BMI'].mean(), inplace = True)
data_df['Age'].fillna(data_df['Age'].mean(), inplace = True)

# Model Building
from sklearn.model_selection import train_test_split

X = data_df.drop(columns='Class (1: positive for diabetes, 0: negative for diabetes)')
y = data_df['Class (1: positive for diabetes, 0: negative for diabetes)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using GridSearchCV to find the best algorithm for this problem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Creating a function to calculate best model for this problem
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'svc': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return print(pd.DataFrame(scores, columns=['model','best_parameters','score']))

find_best_model(X_train, y_train)


# Using cross_val_score for gaining average accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0), X_train, y_train, cv=5)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))

classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
scfilename = 'standardscalar.pickle'
pickle.dump(sc, open(scfilename, 'wb'))
pickle.dump(classifier, open(filename, 'wb'))