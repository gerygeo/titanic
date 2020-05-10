import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.metrics import *
from scipy.stats import mode


def get_title(names):
    titles_map = {"Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
                  }
    titles = names.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = titles.map(titles_map)

    return titles


def get_data(filename):
    df = pd.read_csv(filename)
    try:
        Y = df['Survived']
        df.drop('Survived', axis=1, inplace=True)
    except:
        Y = pd.DataFrame()
        Y['PassengerId'] = df['PassengerId']

    df.drop('PassengerId', axis=1, inplace=True)

    #Age
    df['Age'].fillna(np.mean(df['Age']), inplace=True)

    #FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    #Title/ Name
    df['Title'] = get_title(df['Name'])
    title_dummies = pd.get_dummies(df['Title'], prefix='Embarked')
    df = pd.concat([df, title_dummies], axis=1)
    df.drop(['Title', 'Name'], axis=1, inplace=True)

    #Cabin
    df['Cabin'] = df['Cabin'].apply(lambda x: 1 if x is not None else 0)

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    df.drop(['Cabin', 'Embarked'], axis=1, inplace=True)

    df.fillna(0, inplace=True)
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
    df.drop('Ticket', axis=1, inplace=True)

    df.head(5).to_csv("training_test.csv", index=False)
    return df[['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare']], Y


def dt_model(X, Y, X_test, Y_test):
    dtc = DecisionTreeClassifier(max_depth=10).fit(X, Y)
    Y_pred = dtc.predict(X)

    Y_test_pred = dtc.predict(X_test)

    Y_test['Survived'] = Y_test_pred[0]

    Y_test.to_csv("submission_dtc.csv", index=False)

    return get_metrics(Y, Y_pred)


def rf_with_cv(X, Y, X_test, Y_test):
    rf = [DecisionTreeClassifier(max_depth=10) for i in range(10)]
    survived_ind = Y[Y == 1].index
    not_survived_ind = Y[Y == 0].index
    for dt in rf:
        s_ind = np.random.choice(survived_ind, size=150, replace=True)
        ns_ind = np.random.choice(not_survived_ind, size=150, replace=True)
        ind = np.concatenate((s_ind, ns_ind))

        dt.fit(X.iloc[ind, :], Y.iloc[ind])
    result = []
    for dt in rf:
        result.append(dt.predict(X))
    Y_pred = mode(result, axis=0)[0][0]

    result_test = []
    for dt in rf:
        result_test.append(dt.predict(X_test))
    Y_test['Survived'] = mode(result_test, axis=0)[0][0]

    Y_test.to_csv("submission_rf_with_cv.csv", index=False)

    return get_metrics(Y, Y_pred)


def rf_model(X, Y, X_test, Y_test):
    params_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [20, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(forest, scoring='accuracy', param_grid=params_grid, cv=cross_validation)

    grid_search.fit(X, Y)
    params = grid_search.best_params_
    
    rfc = RandomForestClassifier(**params).fit(X, Y)

    Y_pred = rfc.predict(X)

    Y_test_pred = rfc.predict(X_test)

    Y_test['Survived'] = Y_test_pred.astype(int)

    Y_test.to_csv("submission_rfc.csv", index=False)

    return get_metrics(Y, Y_pred)


def bc_model(X, Y, X_test, Y_test):

    bc = BaggingClassifier(max_samples=20).fit(X, Y)
    Y_pred = bc.predict(X)

    Y_test['Survived'] = bc.predict(X_test)

    Y_test.to_csv("submission_bc.csv", index=False)

    return get_metrics(Y, Y_pred)


def knn_model(X, Y, X_test, Y_test):

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(X, Y)

    Y_pred = knn.predict(X)

    Y_test_pred = knn.predict(X_test)

    Y_test['Survived'] = Y_test_pred

    Y_test.to_csv("submission_kt.csv", index=False)
    return get_metrics(Y, Y_pred)


def get_metrics(Y_true, Y_pred):
    accuracy = accuracy_score(Y_true, Y_pred)
    # auc
    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
    auc_sc = auc(fpr, tpr)
    # Precision
    precision = precision_score(Y_true, Y_pred)
    # Recall
    recall = recall_score(Y_true, Y_pred)
    # f1-score
    f1score = f1_score(Y_true, Y_pred)

    return accuracy, auc_sc, precision, recall, f1score




if __name__ == '__main__':
    X, Y = get_data("train.csv")
    X_test, Y_test = get_data("test.csv")

    print(dt_model(X, Y, X_test, Y_test))
    print(rf_with_cv(X, Y, X_test, Y_test))
    # Model with best prediction: accuracy of 0.76076
    print(rf_model(X, Y, X_test, Y_test))
    print(bc_model(X, Y, X_test, Y_test))
    print(knn_model(X, Y, X_test, Y_test))