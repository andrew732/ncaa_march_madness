'''
 Runs stuff
'''

import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import numpy as np

def combine_tables():
    root_dir = os.getcwd() + '/Merged'
    files = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]
    files.sort()
    files.remove('2017.xlsx')
    df1 = pd.read_excel('Merged/2017.xlsx')
    for file in files:
        if file != ".DS_Store":
            df2 = pd.read_excel('Merged/' + file)
            frames = [df1, df2]
            df1 = pd.concat(frames)
    df1 = df1.dropna(axis=0) # DROPPING MISSING DATA
    writer = pd.ExcelWriter('Data/combined.xlsx')
    df1.to_excel(writer, 'Sheet1')
    writer.save()
    return df1


# drop school and extracts labels
def clear_tables(df):
    df = df.drop(['School'], axis=1)
    return df['Games'], df.drop(['Games'], axis=1)


def simple_model(data, labels):
    models = []
    models.append(('Linear Regression', LinearRegression()))
    models.append(('Ridge', Ridge()))
    models.append(('Lasso', Lasso()))

    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
        model.fit(X_train, y_train)
        # Root mean squared error
        print('%s %s Root mean squared error: %.2f' % ("", "",
                                                       (np.mean((model.predict(X_test) - y_test) * 2)) * 0.5))
        # R squared value
        print('%s %s R squared value: %.2f' % ("", "", model.score(X_test, y_test)))

data = combine_tables()
labels,data = clear_tables(data)
simple_model(data, labels)