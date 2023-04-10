# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

def report(model, X, y, k_fold=10):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=False)
    roc_auc_scores = []
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    confusion_matrixs = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc_scores.append(roc_auc_score(y_test, y_pred))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        confusion_matrixs.append(confusion_matrix(y_test, y_pred))
        
    roc_auc_scores = np.mean(roc_auc_scores)
    accuracy_scores = np.mean(accuracy_scores)
    f1_scores = np.mean(f1_scores)
    recall_scores = np.mean(recall_scores)
    precision_scores = np.mean(precision_scores)
    confusion_matrixs = np.mean(confusion_matrixs, axis=0)
    df_results = pd.DataFrame({'Roc auc': roc_auc_scores, 'Accuracy': accuracy_scores, 'F1': f1_scores, 'Recall': recall_scores, 'Precision': precision_scores}, index=[model.__class__.__name__])
    df_confusion_matrix = pd.DataFrame(confusion_matrixs, columns=['Predicted 0', 'Predicted 1'], index=['Real 0', 'Real 1'])

    if int(confusion_matrixs[0][0]) + int(confusion_matrixs[1][0]) == 0:
        precision_0 = 0
    else:
        precision_0 = int(confusion_matrixs[0][0]) / (int(confusion_matrixs[0][0]) + int(confusion_matrixs[1][0]))
    
    if int(confusion_matrixs[1][1]) + int(confusion_matrixs[0][1]) == 0:
        precision_1 = 0
    else:
        precision_1 = int(confusion_matrixs[1][1]) / (int(confusion_matrixs[1][1]) + int(confusion_matrixs[0][1]))
    
    if int(confusion_matrixs[0][0]) + int(confusion_matrixs[0][1]) == 0:
        recall_0 = 0
    else:
        recall_0 = int(confusion_matrixs[0][0]) / (int(confusion_matrixs[0][0]) + int(confusion_matrixs[0][1]))

    if int(confusion_matrixs[1][1]) + int(confusion_matrixs[1][0]) == 0:
        recall_1 = 0
    else:
        recall_1 = int(confusion_matrixs[1][1]) / (int(confusion_matrixs[1][1]) + int(confusion_matrixs[1][0]))

    df_confusion_matrix['Precision'] = [precision_0, precision_1]
    df_confusion_matrix['Recall'] = [recall_0, recall_1]

    model.fit(X, y)
    return df_results, df_confusion_matrix

def predict_and_save(model, X_test, test_IDs, file_name):
    Y_test = model.predict_proba(X_test)[:, 1]
    Y_test = np.round(Y_test, 1)
    df = pd.DataFrame({'Id': test_IDs, 'Predicted': Y_test})
    df.to_csv(file_name, index=False)

def plot_roc_curve(model, X, y, k_fold=3):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model.__class__.__name__} ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    model.fit(X, y)

def plot_learning_curve(model, X, y, k_fold=3):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=False)
    train_sizes = np.linspace(0.2, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=skf, n_jobs=-1, train_sizes=train_sizes, scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.title(f'{model.__class__.__name__} Curva de Aprendizado')
    plt.xlabel('Amostras de treino')
    plt.ylabel('F1 score')
    plt.show()