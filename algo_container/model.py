import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from statsmodels.tools import categorical
import joblib
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

def get_data():
    print('\n \n 1. Etape de chargement des donnees')
    fraudes = pd.read_csv('Echan.csv', index_col=0)
    fraudes.reset_index(drop=True, inplace=True)
    print(' \n data avec {0} lignes et {1} colones'.format(
        fraudes.shape[0], fraudes.shape[1]))

    print('\n \n 2. Etape de processing')
    a = np.array(fraudes['type'])
    b = categorical(a, drop=True)
    fraudes['Type_num'] = b.argmax(1)

    fraudes.drop(['step', 'type', 'Clt_Origine', 'Clt_Destinataire',
                'Type_Ori', 'Type_Des'], axis=1, inplace=True)


    # Undersampled dataset
    X_undersample = fraudes.loc[:, fraudes.columns != 'Fraude']
    y_undersample = fraudes.loc[:, fraudes.columns == 'Fraude']
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
        X_undersample, y_undersample, test_size=0.3, random_state=0)
    print("")
    print("Number transactions train dataset: ",
        format(len(X_train_undersample), ',d'))
    print("Number transactions test dataset: ",
        format(len(X_test_undersample), ',d'))

    print('\n \n \n3. Etape de machine learning')

    # Arbre de décision
    clf = DecisionTreeClassifier(max_depth=2,
                                random_state=0)

    clf.fit(X_train_undersample, y_train_undersample)

    # prédire les étiquettes des données invisibles (test)
    y_pred_Arbre = clf.predict(X_test_undersample)

    # Mesurer les performances du modèle
    # La méthode du score renvoie la précision du modèle
    score = clf.score(X_test_undersample, y_test_undersample)

    # Mesurer les performances du modèle
    cnf_matrix_arbre = confusion_matrix(y_test_undersample, y_pred_Arbre)
    cnf_matrix_arbre

    print('\n \n \n 4. Model')
    dump(clf, 'fraudes.joblib')
    clf = joblib.load('fraudes.joblib')

    e = {
        "a": round(accuracy_score(y_test_undersample, y_pred_Arbre), 2),
        "b": round(precision_score(y_test_undersample, y_pred_Arbre), 2),
        "c": round(recall_score(y_test_undersample, y_pred_Arbre), 2),
        "d": cnf_matrix_arbre.tolist()
    }

    return e
