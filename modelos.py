from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

MODELOS = ["svm_linear", "svm_rbf", "logreg", "knn"]

NOMES = {}
NOMES["trivial"] = "Trivial (classe majoritária)"
NOMES["svm_linear"] = "SVM linear"
NOMES["svm_rbf"] = "SVM RBF"
NOMES["logreg"] = "Reg. logística"
NOMES["knn"] = "k-NN"

GRADES = {}
GRADES["svm_linear"] = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}
GRADES["svm_rbf"] = {"clf__C": [0.1, 1, 10], "clf__gamma": [0.001, 0.01, 0.1, "scale"]}
GRADES["logreg"] = {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
GRADES["knn"] = {"clf__n_neighbors": [1, 3, 5, 11, 21, 31, 51, 101], "clf__weights": ["uniform", "distance"]}

def criar_modelo(nome):
    if nome == "svm_linear":
        return SVC(kernel="linear", class_weight="balanced", cache_size=1000)
    if nome == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", cache_size=1000)
    if nome == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=5000)
    if nome == "knn":
        return KNeighborsClassifier()

def criar_pipeline(nome):
    return Pipeline([("scaler", StandardScaler()), ("clf", criar_modelo(nome))])
