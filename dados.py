import pandas as pd
from sklearn.model_selection import train_test_split

SEMENTE = 42

ARQ_CDC = "data/cdc_diabetes.arff"
ARQ_ESTUDANTES = "data/dataset.csv"

def ler_cdc():
    arq = open(ARQ_CDC, encoding="utf-8", errors="replace")
    linhas = arq.readlines()
    arq.close()

    colunas = []
    inicio_dados = 0
    i = 0
    while i < len(linhas):
        l = linhas[i].strip()
        if l.lower().startswith("@attribute"):
            colunas.append(l.split()[1])
        if l.lower().startswith("@data"):
            inicio_dados = i + 1
            break
        i += 1

    tabela = []
    i = inicio_dados
    while i < len(linhas):
        l = linhas[i].strip()
        if l != "" and l[0] != "%":
            valores = l.split(",")
            linha = []
            for v in valores:
                linha.append(float(v))
            tabela.append(linha)
        i += 1

    df = pd.DataFrame(tabela, columns=colunas)
    y = df["Diabetes_binary"].astype(int)
    X = df.drop(columns=["Diabetes_binary"])
    return X, y

def ler_estudantes():
    df = pd.read_csv(ARQ_ESTUDANTES, encoding="utf-8-sig")
    novas = []
    for c in df.columns:
        novas.append(c.strip().replace("\t", ""))
    df.columns = novas

    y = []
    for alvo in df["Target"]:
        if alvo.strip() == "Dropout":
            y.append(1)
        else:
            y.append(0)
    y = pd.Series(y)
    X = df.drop(columns=["Target"]).astype(float)
    return X, y

def subamostra(X, y, n):
    if n is None or n >= len(X):
        return X, y
    Xs, lixo1, ys, lixo2 = train_test_split(X, y, train_size=n, stratify=y, random_state=SEMENTE)
    return Xs, ys
