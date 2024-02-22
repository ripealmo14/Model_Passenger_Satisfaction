# Código de Entrenamiento - Modelo SP
############################################################################

import numpy as np 
import pandas as pd
import pickle
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


def train_model(filename):
    """Train model
    """
    df = pd.read_csv("../data/processed/data_train.csv")
    y = df['satisfaction']
    X = df.drop(['satisfaction'],axis=1)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = 36000, test_size = 4000,
                                                    stratify = y, random_state = 2023)

    data_exporting(y_test, "target_validate.csv")
    data_exporting(X_test, "features_validate.csv")

    # Parameters
    alpha_local_opt = 5.623413251903491
    hidden_layer_local_opt_1 = 40
    hidden_layer_local_opt_2 = 21

    # Se corre el modelo con los hiperparametros optimos y se corre con la data para ver los resultados estadisticos del modelo.
    mlp_model = MLPClassifier(alpha = alpha_local_opt,
                            hidden_layer_sizes = (hidden_layer_local_opt_1, hidden_layer_local_opt_2),
                            solver = 'lbfgs',
                            max_iter = 1000,
                            activation = 'logistic',
                            random_state = 42)

    mlp_model.fit(X_train, y_train)

    # Save Best Model
    filename = '../models/best_model.pkl'
    pickle.dump(mlp_model, open(filename, 'wb'))


def data_exporting(df, filename):
    """Data export
    """
    df.to_csv(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')


def main():
    train_model('credit_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
