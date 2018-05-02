# takes a model tests with different filling types
def diff_filling_in_data(model):
    drop_types = ["drop", "mean"]
    accuracies = []
    for mtype in drop_types:
        data = get_all_features(mtype)
        labels, data = split_data_labels(data)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        print(accuracy)
        accuracies.append(accuracy)

    return accuracies


def try_different_combinations_of_data(model):
    data_sources = []
    data_sources.append(get_all_features("mean"))
    data_sources.append(get_pure_basic_features("drop"))

    accuracies = []
    for data_source in data_sources:
        labels, data = split_data_labels(data_source)

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