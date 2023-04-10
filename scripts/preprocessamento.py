# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import pandas as pd
import numpy as np
import os

from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2

def normalizar(df, atributos):
    df[atributos] = (df[atributos] - df[atributos].mean()) / df[atributos].std()
    return df

def selecionar_atributos(df, atributos):
    atributos = [atributo for atributo in atributos if atributo in df.columns]
    df = df[atributos]
    return df

def tratar_faltantes(df, n_neighbors=12):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns , index=df.index)    
    return df

def tratar_atributos_categoricos(df):
    df = pd.get_dummies(df)
    return df

def remover_atributos(df, atributos):
    df = df.drop(atributos, axis=1)
    return df

def selectKBest(df, Y, n_features):
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(df, Y)
    f_values = selector.scores_
    df = df.iloc[:, selector.get_support()]
    f_values = f_values[selector.get_support()]
    return df, f_values

def selectKBestChi2(df, Y, n_features):
    selector = SelectKBest(chi2, k=n_features)
    selector.fit(df, Y)
    p_values = selector.pvalues_
    df = df.iloc[:, selector.get_support()]
    p_values = p_values[selector.get_support()]
    return df, p_values

def aumentar_peso_atributos(df, atributos, peso):
    df[atributos] = df[atributos] * peso
    return df