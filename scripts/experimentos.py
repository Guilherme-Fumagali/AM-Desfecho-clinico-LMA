# Arquivo com todas as funcoes e codigos referentes aos experimentos

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

def grid_search(model, X, y, params, k_fold=10):
    grid = GridSearchCV(model, params, cv=k_fold, scoring='roc_auc', n_jobs=-1)
    grid.fit(X, y)
    roc_auc_scores = grid.cv_results_['mean_test_score'].max()
    return grid.best_params_, roc_auc_scores

def init_model(model, X, y, params, k_fold=10):
    best_params, best_score = grid_search(model, X, y, params, k_fold)
    if 'n_jobs' in model.get_params():
        model = model.set_params(**best_params, n_jobs=-1)
    else:
        model = model.set_params(**best_params)
    model.fit(X, y)
    return model, best_params, best_score

def print_scores(model, best_params, score):
    print(f'Modelo: {model}')
    print(f'Score: {score}')
    print(f'Melhores parametros: {best_params}')
    print()

def init_model_with_feature_selection(model, X, y, params, k_fold=10):
    best_params, best_score = grid_search(model, X, y, params, k_fold)
    if 'n_jobs' in model.get_params():
        model = model.set_params(**best_params, n_jobs=-1)
    else:
        model = model.set_params(**best_params)
    rfecv = RFECV(estimator=model, step=1, cv=k_fold, scoring='roc_auc')
    rfecv.fit(X, y)
    model.fit(X[:, rfecv.support_], y)
    return model, best_params, best_score, rfecv