# Script de Preparación de Datos
###################################

import warnings
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore")


def read_file_csv(filename):
    """ Leemos los archivos csv
    """
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


def data_preparation(df, type_df):
    # Eliminar las primeras dos columnas con data innecesaria
    df.drop(df.iloc[:, [0,1]], axis=1, inplace=True)

    #  Columnas con valores categoricos pueden ser del tipo 'object' o 'int64'
    categorical_indexes = [0, 1, 3, 4] + list(range(6,20))
    df.iloc[:, categorical_indexes] = df.iloc[:, categorical_indexes].astype('category')

    # Como el porcentaje de los calores vacios es menor al 1% podriamos eliminar o reemplaxar por la mediana estos valores
    df = df.dropna()

    # Dividir la data en numerica y categorica
    numerical_columns = [c for c in df.columns if df[c].dtype.name != 'category']
    numerical_columns.remove('satisfaction')
    categorical_columns = [c for c in df.columns if df[c].dtype.name == 'category']
    df_describe = df.describe(include = ['category'])

    # Divir las columnas categoricas en binarias y no binarias
    binary_columns = [c for c in categorical_columns if df_describe[c]['unique'] == 2]
    nonbinary_columns = [c for c in categorical_columns if df_describe[c]['unique'] > 2]

    # Codificación
    # original_df = df[binary_columns].copy()
    df[binary_columns] = df[binary_columns].astype('category').apply(lambda x: x.cat.codes)
    df[binary_columns] = df[binary_columns].astype('category')

    df_nonbinary = pd.get_dummies(df[nonbinary_columns])
    df_numerical = df[numerical_columns]
    df_numerical = (df_numerical - df_numerical.mean(axis=0))/df_numerical.std(axis=0)
    df_final = pd.concat((df_numerical,
                          df_nonbinary,
                          df[binary_columns],
                          df['satisfaction']),
                          axis=1)

    if type_df == "train":
        feature_nonbinary_columns = df_final.columns.to_list()
        df_columns = pd.DataFrame(feature_nonbinary_columns, columns=["Columns"])
        data_exporting(df_columns, "columns_train.csv")
    elif type_df == "score":
        columns_all = pd.read_csv("../data/processed/columns_train.csv")
        columns_now = df_final.columns.to_list()
        for column in columns_all["Columns"]:
            if column not in columns_now:
                df_final[column] = 0

        columns = columns_all["Columns"].to_list()
        df_final = df_final[columns]
        df_final = df_final.drop(['satisfaction'], axis=1)

    return df_final


def data_exporting(df, filename):
    """Data escport
    """
    df.to_csv(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')


def main():
    """main
    """
    df1 = read_file_csv("data.csv")
    tdf1 = data_preparation(df1, "train")
    data_exporting(tdf1, "data_train.csv")

    df2 = read_file_csv("data_score.csv")
    tdf2 = data_preparation(df2, "score")
    data_exporting(tdf2, "data_score.csv")


if __name__ == "__main__":
    main()
