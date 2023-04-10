# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def importar_dados(FILES_DIRECTORY):
    def to_float(x):
        try:
            if isinstance(x, str):
                return np.float64(x.replace(',', '.'))
            else:
                return np.float64(x)
        except:
            return np.nan
        
    def str_treatment(x):
        if isinstance(x, str):
            return x.capitalize()
        else:
            return x
    
    train_class_ids = pd.read_csv(os.path.join(FILES_DIRECTORY, 'train.csv'), index_col='Sample ID')
    test_ids = pd.read_csv(os.path.join(FILES_DIRECTORY, 'test.csv'), index_col='Sample ID')

    df_clinical_data = pd.read_csv(os.path.join(FILES_DIRECTORY, 'clinical_data.csv'), index_col='Sample ID', converters={'Bone Marrow Blast Percentage': to_float})
    df_clinical_data = df_clinical_data.applymap(str_treatment)

    df_genetic_expression_data = pd.read_csv(os.path.join(FILES_DIRECTORY, 'genetic_expression_data.csv'), index_col='Sample ID')
    df_genetic_expression_data.columns = [f'{col}_expression' for col in df_genetic_expression_data.columns]

    df_genetic_mutation_data = pd.read_csv(os.path.join(FILES_DIRECTORY, 'genetic_mutation_data.csv'), index_col='Sample ID')
    df_genetic_mutation_data.columns = [f'{col}_mutation' for col in df_genetic_mutation_data.columns]
    
    df = pd.merge(df_clinical_data, df_genetic_expression_data, on='Sample ID', how='outer')
    df = pd.merge(df, df_genetic_mutation_data, on='Sample ID', how='outer')
    df = df.drop_duplicates()

    coluns_names = {'clinical': df_clinical_data.columns, 'genetic_expression': df_genetic_expression_data.columns, 'genetic_mutation': df_genetic_mutation_data.columns}
    return df, train_class_ids, test_ids, coluns_names

def boxplot(x, y, title=None, xlabel='X', ylabel='Y', ax=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.boxplot(x=x, y=y, ax=ax)

def matriz_de_correlacao(x, title='Matriz de Correlacao', size=(10, 6)):
    plt.figure(figsize=size)
    plt.title(title)
    sns.heatmap(x.corr(), cmap='coolwarm', annot=True)
    plt.show()

def histograma(x, y, title=None, xlabel='X', ax=None):
    plt.title(title)
    plt.xlabel(xlabel)
    sns.histplot(x=x, hue=y, ax=ax, multiple='stack')

def grafico_de_barras(x, y, title=None, ax=None):
    plt.title(title)
    sns.countplot(x=x, hue=y, ax=ax) 
    if ax is not None:
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')

def grafico_de_dispersao(x1, x2, y, title=None, xlabel='X', ylabel='Y', ax=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.scatterplot(x=x1, y=x2, hue=y, ax=ax)

def tabela_de_contingencia(x, y):
    tab = pd.crosstab(x, y, normalize='index') 
    tab['count'] = pd.crosstab(x, y).apply(sum, axis=1)
    return tab