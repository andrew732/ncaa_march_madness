'''
 Runs stuff
'''

import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np


# gets all features with filling method
def combine_tables(method):
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


def get_specific_year(year):
    df1 = pd.read_excel('Merged/' + year + '.xlsx')
    df1 = df1.dropna(axis=0)  # DROPPING MISSING DATA
    return df1


# drop school and extracts labels
def clear_tables(df):
    df = df.drop(['School'], axis=1)
    return df['Games'], df.drop(['Games'], axis=1)


def simple_regression(data, labels):
    models = []
    models.append(('Linear Regression', LinearRegression()))
    models.append(('Ridge', Ridge()))
    models.append(('Lasso', Lasso()))
    accuracies = []

    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
        model.fit(X_train, y_train)
        # Root mean squared error
        print('%s %s Root mean squared error: %.2f' % ("", "",
                                                       (np.mean((model.predict(X_test) - y_test) * 2)) * 0.5))
        # R squared value
        rsq = model.score(X_test, y_test)
        print('%s %s R squared value: %.2f' % ("", "", rsq))
        accuracies.append(rsq)
    max = np.argmax(accuracies)
    return models[max]


def simple_classification(data, labels):
    models = []
    # models.append(('Linear SVM', LinearSVC()))
    models.append(('Linear SGD', SGDClassifier(random_state=0, learning_rate="invscaling", loss="log", penalty="l1",
                                               max_iter=1500, alpha=.0001, eta0=1.0, epsilon=.0001)))
    models.append(('Naive Bayes', GaussianNB()))

    accuracies = []

    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(name)
        accuracy = accuracy_score(y_test, prediction)
        print(accuracy)
        accuracies.append(accuracy)
    max = np.argmax(accuracies)
    return models[max]


# takes a model tests with different filling types
def diff_filling_in_data(model):
    drop_types = ["drop", "mean"]
    accuracies = []
    for mtype in drop_types:
        data = combine_tables(mtype)
        labels, data = clear_tables(data)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        print(accuracy)
        accuracies.append(accuracy)

    return accuracies


def try_different_combinations_of_data(model):
    data_sources = []
    data_sources.append(combine_tables("mean"))
    data_sources.append(get_pure_basic_features("drop"))

    accuracies = []
    for data_source in data_sources:
        labels, data = clear_tables(data_source)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        print(accuracy)
        accuracies.append(accuracy)

        # to work with advanced data (since it does not have labels)
        # data_adv = get_pure_advanced("mean").drop(['School'], axis=1)
        # labels, idc = clear_tables(data_sources[0]) # hack to get labels
        # X_train, X_test, y_train, y_test = train_test_split(data_adv, labels, test_size=0.4, random_state=0)
        # model.fit(X_train, y_train)
        # prediction = model.predict(X_test)
        # accuracy = accuracy_score(y_test, prediction)
        # print(accuracy)
        # accuracies.append(accuracy)


# return espn score like system
def get_score(prediction, actual):
    rounds = [10, 30, 70, 150, 310, 630] # cummulative
    if actual == 0:
        if prediction == 0:
            return 10
        else:
            return 0
    else:
        if prediction == 0:
            return 0
        elif prediction > actual:
            rounds_correct = actual + 1
            return rounds[rounds_correct]
        else:
            rounds_correct = prediction - 1
            return rounds[rounds_correct]


# given the actual bracket array. What was the score
def get_actual_score(actual):
    rounds = [10, 30, 70, 150, 310, 630]
    score = 0
    for i in actual:
        score = score + rounds[i]
    return score


# takes a trained model and predicts on a year. Returns
def simulated_bracket_score(model, year_data, labels):
    year_data = year_data.drop(['School'], axis=1).drop(['Games'], axis=1)
    predictions = model.predict(year_data)
    score = 0
    for i in range(0, len(predictions)):
        score = score + get_score(predictions[i], labels[i])
    print("Predicted Score: " + str(score))

    actual = get_actual_score(labels)
    print("Actual Score: " + str(actual))

    return score


data = combine_tables("drop")
labels, data = clear_tables(data)
data_2017 = get_specific_year('2017')
labels2017, data2017 = clear_tables(data_2017)
name, model = simple_classification(data, labels)
simulated_bracket_score(model, data_2017, labels2017.as_matrix())
# model2 = SGDClassifier(random_state=0, learning_rate="invscaling", loss="log", penalty="l1",
#                        max_iter=1500, alpha=.0001, eta0=1.0, epsilon=.0001)
# diff_filling_in_data(model2)
# simple_model(data, labels)
# try_different_combinations_of_data(model2)
