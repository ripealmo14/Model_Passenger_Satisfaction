# Código de Evaluación
#######################

import pandas as pd
import pickle
import os
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, \
                            accuracy_score, \
                            f1_score, \
                            precision_score, \
                            recall_score, \
                            classification_report, \
                            confusion_matrix

warnings.filterwarnings("ignore")


def eval_model(file_x_test, file_y_test):
    """eval model
    """
    y_test = pd.read_csv(os.path.join('../data/processed/', file_y_test))
    print(file_y_test, ' cargado correctamente')
    X_test = pd.read_csv(os.path.join('../data/processed/', file_x_test))
    print(file_x_test, ' cargado correctamente')
    features = X_test.columns.tolist()
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación
    X_test['probability']  = model.predict_proba(X_test[features])[:, 1]
    X_test['prediction']  = model.predict(X_test[features])
    # Generamos métricas de diagnóstico
    metricsRfc = pd.DataFrame({'metric':['AUC', 'Gini', 'Accuracy', 'Precision', 'Recall', 'F1-score'],
                                'nn_test':[roc_auc_score(y_test, X_test.probability),
                                        (roc_auc_score(y_test, X_test.probability)*2-1),
                                        accuracy_score(y_test, X_test.prediction),
                                        precision_score(y_test, X_test.prediction, pos_label='satisfied'),
                                        recall_score(y_test, X_test.prediction, pos_label='satisfied'),
                                        f1_score(y_test, X_test.prediction, pos_label='satisfied')]})
    print(confusion_matrix(y_test, X_test.prediction))
    print(metricsRfc)


def main():
    """main
    """
    eval_model('features_validate.csv', 'target_validate.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()