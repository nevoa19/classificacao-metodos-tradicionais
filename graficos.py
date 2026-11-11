import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dados import SEMENTE, subamostra
from modelos import MODELOS, NOMES, criar_modelo
from tabelas import numero_br

PASTA_FIG = "resultados/figuras"

AZUL = '#0275d8'
VERMELHO = '#d9534f'

def salvar_figura(nome):
    if not os.path.exists(PASTA_FIG):
        os.makedirs(PASTA_FIG)
    plt.savefig(PASTA_FIG + "/" + nome + ".pdf", bbox_inches="tight")
    plt.savefig(PASTA_FIG + "/" + nome + ".png", dpi=200, bbox_inches="tight")
    plt.close()

def grafico_distribuicoes(X, y, colunas, tag, classes):
    fig, eixos = plt.subplots(1, 3, figsize=(7, 2), dpi=150)
    i = 0
    while i < 3:
        col = colunas[i]
        eixos[i].hist(X.loc[y == 0, col], bins=30, alpha=0.6, density=True, color=AZUL, label=classes[0])
        eixos[i].hist(X.loc[y == 1, col], bins=30, alpha=0.6, density=True, color=VERMELHO, label=classes[1])
        eixos[i].set_title(col, fontsize=8)
        eixos[i].tick_params(labelsize=7)
        i += 1
    eixos[0].legend(frameon=False, fontsize=7)
    salvar_figura(tag + "_distribuicoes")

def projetar_2d(X, y, n_pontos):
    Xs, ys = subamostra(X, y, n_pontos)
    Z = StandardScaler().fit_transform(Xs)
    pca = PCA(n_components=2, random_state=SEMENTE).fit(Z)
    variancia = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    return pca.transform(Z), np.array(ys), variancia

def grafico_pca(X, y, tag, classes):
    Z, ys, variancia = projetar_2d(X, y, 3000)
    plt.figure(figsize=(3.4, 2.6), dpi=150)
    plt.scatter(Z[ys == 0, 0], Z[ys == 0, 1], s=4, alpha=0.5, c=AZUL, label=classes[0])
    plt.scatter(Z[ys == 1, 0], Z[ys == 1, 1], s=4, alpha=0.5, c=VERMELHO, label=classes[1])
    plt.xlabel("PC1", fontsize=8)
    plt.ylabel("PC2", fontsize=8)
    plt.title(f"PCA (variancia explicada: {variancia * 100:.1f}%)", fontsize=9, fontweight='bold')
    plt.legend(frameon=False, markerscale=3, fontsize=7)
    salvar_figura(tag + "_pca")
    return float(variancia)

def grafico_matrizes(resultados, tag, classes):
    fig, eixos = plt.subplots(1, len(MODELOS), figsize=(1.9 * len(MODELOS), 2.0), dpi=150, sharey=True)
    fig.subplots_adjust(wspace=0.15)
    j = 0
    while j < len(MODELOS):
        m = MODELOS[j]
        cm = resultados[m]["teste"]["confusao"]
        norm = []
        a = 0
        while a < 2:
            total_linha = cm[a][0] + cm[a][1]
            norm.append([cm[a][0] / total_linha, cm[a][1] / total_linha])
            a += 1
        eixos[j].imshow(norm, cmap="Blues", vmin=0, vmax=1)
        a = 0
        while a < 2:
            b = 0
            while b < 2:
                cor = "black"
                if norm[a][b] > 0.5:
                    cor = "white"
                eixos[j].text(b, a, numero_br(cm[a][b]), ha="center", va="center", color=cor, fontsize=7)
                b += 1
            a += 1
        eixos[j].set_xticks([0, 1], classes, fontsize=6)
        if j == 0:
            eixos[j].set_yticks([0, 1], classes, fontsize=6)
            eixos[j].set_ylabel("real", fontsize=7)
        eixos[j].set_title(NOMES[m], fontsize=9)
        eixos[j].set_xlabel("previsto", fontsize=7)
        j += 1
    salvar_figura(tag + "_matrizes_confusao")

def grafico_fronteiras(X, y, melhores, tag, classes):
    Z, ys, variancia = projetar_2d(X, y, 4000)
    x_min = Z[:, 0].min() - 0.5
    x_max = Z[:, 0].max() + 0.5
    y_min = Z[:, 1].min() - 0.5
    y_max = Z[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grade = []
    lx = xx.flatten()
    ly = yy.flatten()
    i = 0
    while i < len(lx):
        grade.append([lx[i], ly[i]])
        i += 1

    gerador = np.random.default_rng(SEMENTE)
    idx0 = np.where(ys == 0)[0]
    idx1 = np.where(ys == 1)[0]
    mostrar0 = gerador.choice(idx0, min(400, len(idx0)), replace=False)
    mostrar1 = gerador.choice(idx1, min(400, len(idx1)), replace=False)

    fig, eixos = plt.subplots(1, len(MODELOS), figsize=(1.9 * len(MODELOS), 2), dpi=150)
    j = 0
    while j < len(MODELOS):
        m = MODELOS[j]
        clf = criar_modelo(m)
        for k in melhores[m]:
            if k == "C":
                clf.C = melhores[m][k]
            if k == "gamma":
                clf.gamma = melhores[m][k]
            if k == "n_neighbors":
                clf.n_neighbors = melhores[m][k]
            if k == "weights":
                clf.weights = melhores[m][k]
        clf.fit(Z, ys)
        regiao = clf.predict(grade).reshape(xx.shape)
        eixos[j].contourf(xx, yy, regiao, cmap=ListedColormap(["#cddcf0", "#f4cdd2"]), alpha=0.9)
        eixos[j].scatter(Z[mostrar0, 0], Z[mostrar0, 1], s=2, c=AZUL, alpha=0.6, label=classes[0])
        eixos[j].scatter(Z[mostrar1, 0], Z[mostrar1, 1], s=2, c=VERMELHO, alpha=0.6, label=classes[1])
        eixos[j].set_title(NOMES[m], fontsize=9)
        eixos[j].set_xticks([])
        eixos[j].set_yticks([])
        j += 1
    eixos[0].set_ylabel(f"PC2 (var. PC1+PC2: {variancia * 100:.0f}%)", fontsize=7)
    eixos[0].legend(frameon=False, markerscale=4, fontsize=6, loc="lower left")
    salvar_figura(tag + "_fronteiras")

def grafico_painel(todos):
    metricas = ["acuracia", "precisao", "revocacao", "f1"]
    rotulos = ['acuracia', 'precisao', 'revocacao', 'F1']
    tags = list(todos.keys())
    fig, eixos = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 2.4), dpi=150, squeeze=False)
    t = 0
    while t < len(tags):
        ax = eixos[0][t]
        res = todos[tags[t]]["modelos"]
        modelos = list(res.keys())
        largura = 0.8 / len(modelos)
        i = 0
        while i < len(modelos):
            valores = []
            for k in metricas:
                valores.append(res[modelos[i]]["teste"][k])
            posicoes = []
            p = 0
            while p < len(metricas):
                posicoes.append(p + i * largura)
                p += 1
            ax.bar(posicoes, valores, largura, label=NOMES[modelos[i]])
            i += 1
        centro = []
        p = 0
        while p < len(metricas):
            centro.append(p + 0.4 - largura / 2)
            p += 1
        ax.set_xticks(centro, rotulos, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(todos[tags[t]]["titulo"], fontsize=9, fontweight='bold')
        ax.grid(True, axis="y", ls="--", alpha=0.5)
        t += 1
    eixos[0][len(tags) - 1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    salvar_figura("painel_comparativo")
