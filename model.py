'''
 Runs stuff
'''

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from mord import OrdinalRidge, LAD


# gets all features with filling method
def get_all_features(method):
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
    if method == "drop":
        df1 = df1.dropna(axis=0)  # DROPPING MISSING DATA
    elif method == "mean":
        df1 = df1.fillna(df1.mean())  # fill with mean
    writer = pd.ExcelWriter('Data/combined.xlsx')
    df1.to_excel(writer, 'Sheet1')
    writer.save()
    return df1


def get_pure_basic_features(method):
    root_dir = os.getcwd() + '/Data/BasicStats2'
    files = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]
    files.sort()
    files.remove('2017.xlsx')
    df1 = pd.read_excel('Data/BasicStats2/2017.xlsx')
    for file in files:
        if file != ".DS_Store":
            df2 = pd.read_excel('Data/BasicStats2/' + file)
            frames = [df1, df2]
            df1 = pd.concat(frames)
    if method == "drop":
        df1 = df1.dropna(axis=1, how='all')
        df1 = df1.dropna(axis=0)  # DROPPING MISSING DATA
    elif method == "mean":
        df1 = df1.fillna(df1.mean())  # fill with mean
    return df1


# # does not work
# def get_pure_advanced(method):
#     root_dir = os.getcwd() + '/Data/Advanced'
#     files = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]
#     files.sort()
#     files.remove('2017.csv')
#     df1 = pd.read_csv('Data/Advanced/2017.csv', encoding='utf-8')
#     df1 = df1[df1["School"].str.contains("NCAA")]
#     for file in files:
#         if file != ".DS_Store":
#             df2 = pd.read_csv('Data/Advanced/' + file, encoding='utf-8')
#             df2 = df2[df2["School"].str.contains("NCAA")]
#             frames = [df1, df2]
#             df1 = pd.concat(frames)
#
#     if method == "drop":
#         df1 = df1.dropna(axis=0)  # DROPPING MISSING DATA
#     elif method == "mean":
#         df1 = df1.fillna(df1.mean())  # fill with mean
#     return df1


def get_specific_year(year, method):
    df1 = pd.read_excel('Merged/' + year + '.xlsx')
    if method == "drop":
        df1 = df1.dropna(axis=0)  # DROPPING MISSING DATA
    elif method == "mean":
        df1 = df1.fillna(df1.mean())  # fill with mean
    return df1


# drop school and extracts labels
def split_data_labels(df):
    df = df.drop(['School'], axis=1)
    df = df.drop(['Rk'], axis=1)
    return df['Games'], df.drop(['Games'], axis=1)


def penalty_score_one(prediction, actual):
    matrix = [
        [10, -10, -30, -70, -150],
        [-10, 30, -20, -60, -140],
        [-30, -20, 70, -40, -120],
        [-70, -60, -40, 150, -80],
        [-150, -140, -120, -80, 310]
    ]
    return matrix[actual][prediction]


# takes a trained model and predicts on a year. Returns
def simulated_bracket_score(model, year_data, year_labels):
    score = 0
    predictions = model.predict(year_data)
    predictions = np.rint(predictions)
    for i in range(0, len(predictions)):
        prediction = int(predictions[i])
        if prediction < 0:
            prediction = 0
        if prediction > 4:
            prediction = 4

        score = score + penalty_score_one(prediction, year_labels[i])
    return score


def simple_regression(method):
    data = get_all_features(method)
    train_labels, train_data = split_data_labels(data)
    data_2017 = get_specific_year('2017', method)
    labels2017, data2017 = split_data_labels(data_2017)

    models = []
    models.append(('Linear Regression', LinearRegression()))
    models.append(('Ridge', Ridge()))
    models.append(('Lasso', Lasso()))
    models.append(('OrdinalRidge', OrdinalRidge()))
    models.append(('LAD', LAD()))
    accuracies = []
    bracket_scores = []

    for name, model in models:
        print(name)
        model.fit(train_data, train_labels)
        predictions = model.predict(data2017)
        # Root mean squared error
        print('%s %s Root mean squared error: %.2f' % ("", "",
                                                       (np.mean((predictions - labels2017) ** 2)) ** 0.5))
        # R squared value
        rsq = model.score(data2017, labels2017)
        print('%s %s R squared value: %.2f' % ("", "", rsq))

        accuracies.append(rsq)

        bracket_score = simulated_bracket_score(model, data2017, labels2017.as_matrix())
        bracket_scores.append(bracket_score)
        print("%s Bracket Score: " + str(bracket_score))

    max = np.argmax(bracket_scores)
    return models[max]


def simple_classification(method):
    data = get_all_features(method)
    train_labels, train_data = split_data_labels(data)
    data_2017 = get_specific_year('2017', method)
    labels2017, data2017 = split_data_labels(data_2017)

    models = []
    models.append(('Linear SVM', LinearSVC()))
    models.append(('Linear SGD', SGDClassifier(random_state=0, learning_rate="invscaling", loss="log", penalty="l1",
                                               max_iter=1500, alpha=.0001, eta0=1.0, epsilon=.0001)))
    models.append(('Naive Bayes', GaussianNB()))

    accuracies = []
    bracket_scores = []
    for name, model in models:
        model.fit(train_data, train_labels)
        prediction = model.predict(data2017)
        print(name)
        accuracy = accuracy_score(labels2017.as_matrix(), prediction)
        print(accuracy)
        accuracies.append(accuracy)
        bracket_score = simulated_bracket_score(model, data2017, labels2017.as_matrix())
        bracket_scores.append(bracket_score)
        print("%s Bracket Score: " + str(bracket_score))
    max = np.argmax(bracket_scores)
    return models[max]


if __name__ == "__main__":
    simple_regression("drop")
    simple_classification("drop")
    simple_regression("mean")
    simple_classification("mean")
