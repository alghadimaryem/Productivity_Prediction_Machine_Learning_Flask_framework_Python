from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier  # notre algo depuis sklearn
from sklearn.naive_bayes import GaussianNB  # notre algo depuis sklearn
from sklearn.svm import SVC  # notre algo depuis sklearn
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("home.html")


@app.route("/randomforest")
def randomforest():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input
    # model = KNeighborsClassifier(n_neighbors=3)
    regressor = RandomForestRegressor()

    # X_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # hyperparameters:

    # nombre des arbres dans random forest
    n_estimators = [int(x) for x in np.linspace(start=20, stop=100, num=10)]
    # nombre des features pour chaque split
    max_features = ['auto', 'sqrt']
    # le nombre max des niveaux dans l'arbre
    max_depth = [int(x) for x in np.linspace(10, 90, num=11)]
    max_depth.append(None)
    # nombre min de samples pour diviser un noeud
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # créer random grid (contient les hyper_parametres)
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # model.fit(X_train, y_train)

    model = RandomizedSearchCV(estimator=regressor, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pickle.dump(model, open("model.pkl", "wb"))
    return render_template("randomforest.html")


@app.route("/predict1", methods=["GET", "POST"])
def predict1():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result1 = "your actual productivity is low ( < 0.8 )"
    else:
        result1 = "your actual productivity is high ( >0.8 )"
    return render_template("result1.html", result1=result1)


@app.route("/logisticreg")
def logisticreg():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input
    logreg = LogisticRegression()  # model logreg
    # séparer les data pour training et testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # hyperparameters
    C = np.linspace(1, 200)
    penalty = ['l1', 'l2']
    hyperparameters = dict(C=C, penalty=penalty)

    lr = RandomizedSearchCV(logreg, hyperparameters, n_iter=100, cv=5, random_state=41)
    lr.fit(X_train, y_train)

    pickle.dump(lr, open('model.pkl', 'wb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("logisticreg.html")


@app.route("/predict2", methods=["GET", "POST"])
def predict2():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result2 = "your actual productivity is low ( < 0.8 )"
    else:
        result2 = "your actual productivity is high ( >0.8 )"
    return render_template("result2.html", result2=result2)


@app.route("/knearest")
def knearest():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input

    # séparer les data pour training et testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    # List Hyperparameters that we want to tune.
    leaf_size = list(range(1, 50, 10))
    n_neighbors = list(range(1, 30, 5))
    p = [1, 2]

    # Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # Create new KNN object
    knn = KNeighborsClassifier()

    # Use GridSearch
    kn = GridSearchCV(knn, hyperparameters, cv=5)

    # Fit the model
    kn.fit(X_train, y_train)

    pickle.dump(kn, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("knearest.html")


@app.route("/predict3", methods=["GET", "POST"])
def predict3():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result3 = "your actual productivity is low ( < 0.8 )"
    else:
        result3 = "your actual productivity is high ( >0.8 )"
    return render_template("result3.html", result3=result3)


@app.route("/gradient")
def gradient():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input

    # séparer les data pour training et testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    gbc = GradientBoostingClassifier()

    # Hyperparameters
    parameters = {
        "n_estimators": [5, 50, 250],
        "max_depth": [1, 3, 5, 7],
        "learning_rate": [0.01, 0.1, 1, 10]
    }

    gbb = RandomizedSearchCV(gbc, parameters, n_iter=100, cv=5)

    gbb.fit(X_train, y_train)

    pickle.dump(gbb, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("gradient.html")


@app.route("/predict4", methods=["GET", "POST"])
def predict4():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result4= "your actual productivity is low ( < 0.8 )"
    else:
        result4 = "your actual productivity is high ( >0.8 )"
    return render_template("result4.html", result4=result4)


@app.route('/decisiontree')
def decisiontree():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input

    # séparer les data pour training et testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    dtc = DecisionTreeClassifier()

    param_dict = {
        "criterion": ['gini', 'entropy'],
        "max_depth": range(1, 10),
        "min_samples_split": range(1, 10),
        "min_samples_leaf": range(1, 5)
    }

    dttc = GridSearchCV(dtc, param_grid=param_dict, cv=5, verbose=1, n_jobs=-1)

    dttc.fit(X_train, y_train)

    pickle.dump(dttc, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("decisiontree.html")


@app.route("/predict5", methods=["GET", "POST"])
def predict5():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result5 = "your actual productivity is low ( < 0.8 )"
    else:
        result5 = "your actual productivity is high ( >0.8 )"
    return render_template("result5.html", result5=result5)


@app.route('/vebayes')
def naivebayes():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input

    # séparer les data pour training et testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    gnb = GaussianNB()

    parameters = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }

    nb = GridSearchCV(gnb, param_grid=parameters, cv=5, verbose=True)

    # Fit the model
    nb.fit(X_train, y_train)

    pickle.dump(nb, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("naivebayes.html")


@app.route("/predict6", methods=["GET", "POST"])
def predict6():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result6 = "your actual productivity is low ( < 0.8 )"
    else:
        result6 = "your actual productivity is high ( >0.8 )"
    return render_template("result6.html", result6=result6)


@app.route('/vectormachine')
def vectormachine():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # séparer les variables dépendantes et indépendantes
    y = dataset['class']  # output or label
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'class'], axis='columns')  # input

    # séparer les data pour training et testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    svm = SVC()  # notre model dans la var svm

    #param_grid = {'C': [0.1, 1, 10],
                  #'gamma': [1, 0.1, 0.01],
                  #'kernel': ['rbf', 'poly']
                  #}

    #svvm = GridSearchCV(svm, param_grid, refit=True, cv=5, verbose=2)

    svm.fit(X_train, y_train)

    pickle.dump(svm, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("vectormachine.html")


@app.route("/predict7", methods=["GET", "POST"])
def predict7():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    actualproductivity = float(request.form['actualproductivity'])
    form_array = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers, actualproductivity]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array)[0]
    if prediction == 0:
        result7 = "your actual productivity is low ( < 0.8 )"
    else:
        result7 = "your actual productivity is high ( >0.8 )"
    return render_template("result7.html", result7=result7)


@app.route("/linearreg")
def linearreg():
    dataset = pd.read_csv("garments_worker_productivity.csv")
    dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


    # y c'est notre output, la valeur qu'on va predicter
    y = dataset['actual_productivity']
    # x (input) les valeurs de l'entrée
    X = dataset.drop(['date', 'quarter', 'department', 'day', 'actual_productivity', 'class'], axis='columns')

    # splicing data (80% training and 20% testing data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # importer l'algo de la regression linéaire depuis la bibliothèque sklearn

    # la variable lr_model est notre LR modèle
    lr_model = LinearRegression()

    # Training the algorithm
    lr_model.fit(X_train, y_train)

    pickle.dump(lr_model, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    return render_template("linearreg.html")


@app.route("/predict8", methods=["GET", "POST"])
def predict8():
    team = float(request.form['team'])
    targetedproductivity = float(request.form['targetedproductivity'])
    smv = float(request.form['smv'])
    wip = float(request.form['wip'])
    overtime = float(request.form['overtime'])
    insentive = float(request.form['insentive'])
    idletime = float(request.form['idletime'])
    idlemen = float(request.form['idlemen'])
    numberofstyle = float(request.form['numberofstyle'])
    nworkers = float(request.form['nworkers'])
    form_array1 = np.array([[team, targetedproductivity, smv, wip, overtime, insentive, idletime, idlemen, numberofstyle,
                            nworkers]])
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(form_array1)[0]
    return render_template("result8.html",  result8=prediction )


if __name__ == "__main__":
    app.run(debug=True)
