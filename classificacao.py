import os
import json
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from dados import SEMENTE, ler_cdc, ler_estudantes, subamostra
from modelos import MODELOS, NOMES, GRADES, criar_pipeline
from metricas import N_BOOTSTRAP, calcular_metricas, intervalo_f1, mcnemar
from tabelas import numero_br, salvar_tabela
from graficos import grafico_distribuicoes, grafico_pca, grafico_matrizes, grafico_fronteiras, grafico_painel

RODAR_CDC = True
RODAR_ESTUDANTES = True

PORC_TESTE = 0.20

def rodar_base(tag, titulo, X, y, classes, colunas_stats, folds, sub_svm, sub_knn, sub_treino):
    print(f"\n=== {titulo}")
    qtd_0 = 0
    qtd_1 = 0
    for v in y:
        if v == 1:
            qtd_1 += 1
        else:
            qtd_0 += 1
    print(f"n={len(X)}  atributos={X.shape[1]}  ausentes={int(X.isna().sum().sum())}  classe 0: {qtd_0}  classe 1: {qtd_1}  acuracia trivial={max(qtd_0, qtd_1) / len(y):.3f}")

    linhas_stats = []
    for c in colunas_stats:
        v = X[c]
        linhas_stats.append([c, numero_br(v.mean(), 2), numero_br(v.median(), 2), numero_br(v.std(), 2),
                             numero_br(v.var(), 2), numero_br(v.min(), 2), numero_br(v.max(), 2)])
    salvar_tabela(tag + "_estatisticas", ["Atributo", "Média", "Mediana", "Desv. pad.", "Variância", "Mín.", "Máx."],
                  linhas_stats, titulo + ": estatísticas descritivas de atributos selecionados.", "tab:" + tag + "_estatisticas")
    grafico_distribuicoes(X, y, colunas_stats, tag, classes)
    var_pca = grafico_pca(X, y, tag, classes)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=PORC_TESTE, stratify=y, random_state=SEMENTE)
    print(f"treino={len(Xtr)}  teste={len(Xte)}")

    resultados = {}
    previsoes = {}
    melhores = {}

    classe_maj = 0
    if ytr.sum() > len(ytr) / 2:
        classe_maj = 1
    chute_treino = []
    i = 0
    while i < len(ytr):
        chute_treino.append(classe_maj)
        i += 1
    chute_teste = []
    i = 0
    while i < len(yte):
        chute_teste.append(classe_maj)
        i += 1
    previsoes["trivial"] = chute_teste
    resultados["trivial"] = {}
    resultados["trivial"]["parametros"] = {}
    resultados["trivial"]["f1_vc"] = None
    resultados["trivial"]["treino"] = calcular_metricas(ytr, chute_treino)
    resultados["trivial"]["teste"] = calcular_metricas(yte, chute_teste)

    for m in MODELOS:
        t0 = time.time()
        n_sub = None
        if m == "svm_linear" or m == "svm_rbf":
            n_sub = sub_svm
        if m == "knn":
            n_sub = sub_knn
        Xs, ys = subamostra(Xtr, ytr, n_sub)

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
        busca = GridSearchCV(criar_pipeline(m), GRADES[m], scoring="f1", cv=cv, n_jobs=-1, refit=True)
        busca.fit(Xs, ys)

        melhores[m] = {}
        for chave in busca.best_params_:
            melhores[m][chave.replace("clf__", "")] = busca.best_params_[chave]

        if m == "knn" and n_sub is not None:
            modelo = criar_pipeline(m)
            modelo.named_steps["clf"].n_neighbors = melhores[m]["n_neighbors"]
            modelo.named_steps["clf"].weights = melhores[m]["weights"]
            modelo.fit(Xtr, ytr)
        else:
            modelo = busca.best_estimator_

        Xav, yav = subamostra(Xtr, ytr, sub_treino)
        previsoes[m] = modelo.predict(Xte)

        r = {}
        r["parametros"] = melhores[m]
        r["f1_vc"] = float(busca.best_score_)
        r["subamostra"] = n_sub
        r["treino"] = calcular_metricas(yav, modelo.predict(Xav))
        r["teste"] = calcular_metricas(yte, previsoes[m])
        r["segundos"] = round(time.time() - t0, 1)
        resultados[m] = r
        print(f"{NOMES[m]:16s} {melhores[m]}  F1 vc={r['f1_vc']:.3f}  F1 treino={r['treino']['f1']:.3f}  F1 teste={r['teste']['f1']:.3f}  acc teste={r['teste']['acuracia']:.3f}  ({r['segundos']}s)")

    melhor = MODELOS[0]
    for m in MODELOS:
        if resultados[m]["teste"]["f1"] > resultados[melhor]["teste"]["f1"]:
            melhor = m

    for m in MODELOS:
        resultados[m]["ic95_f1"] = intervalo_f1(yte, previsoes[m])
        if m == melhor:
            resultados[m]["p_mcnemar"] = None
        else:
            resultados[m]["p_mcnemar"] = mcnemar(yte, previsoes[melhor], previsoes[m])

    linhas = []
    for m in MODELOS:
        texto_params = ""
        for k in melhores[m]:
            if texto_params != "":
                texto_params = texto_params + ", "
            texto_params = texto_params + k + "=" + numero_br(melhores[m][k], 4)
        linhas.append([NOMES[m], texto_params, numero_br(resultados[m]["f1_vc"])])
    salvar_tabela(tag + "_hiperparametros", ["Modelo", "Hiperparâmetros", "F1 (VC)"], linhas,
                  titulo + ": hiperparâmetros escolhidos e F1 médio na validação cruzada.", "tab:" + tag + "_hiper", "llc")

    linhas = []
    for m in ["trivial"] + MODELOS:
        t = resultados[m]["teste"]
        linhas.append([NOMES[m], numero_br(t["acuracia"]), numero_br(t["precisao"]), numero_br(t["revocacao"]),
                       numero_br(t["f1"]), numero_br(resultados[m]["treino"]["f1"])])
    salvar_tabela(tag + "_metricas", ["Modelo", "Acur.", "Prec.", "Rev.", "F1", "F1 treino"], linhas,
                  titulo + ": métricas no conjunto de teste (classe positiva: " + classes[1].lower() + ").", "tab:" + tag + "_metricas")

    linhas = []
    for m in MODELOS:
        ic = resultados[m]["ic95_f1"]
        p = resultados[m]["p_mcnemar"]
        if p is None:
            texto_p = "--"
        elif p < 0.0001:
            texto_p = "$<10^{-4}$"
        else:
            texto_p = numero_br(p, 4)
        linhas.append([NOMES[m], "[" + numero_br(ic[0]) + "; " + numero_br(ic[1]) + "]", texto_p])
    salvar_tabela(tag + "_significancia", ["Modelo", "IC 95\\% do F1", "$p$ McNemar"], linhas,
                  titulo + ": IC 95\\% do F1 (bootstrap, " + str(N_BOOTSTRAP) + " reamostragens) e teste de McNemar contra o melhor modelo (" + NOMES[melhor] + ").",
                  "tab:" + tag + "_sig")

    grafico_matrizes(resultados, tag, classes)
    grafico_fronteiras(X, y, melhores, tag, classes)

    saida = {}
    saida["titulo"] = titulo
    saida["n"] = len(X)
    saida["variancia_pca"] = var_pca
    saida["n_treino"] = len(Xtr)
    saida["n_teste"] = len(Xte)
    saida["melhor_modelo"] = melhor
    saida["modelos"] = resultados
    return saida

if __name__ == "__main__":
    if not os.path.exists("resultados"):
        os.makedirs("resultados")

    todos = {}
    if os.path.exists("resultados/resultados.json"):
        arq = open("resultados/resultados.json", encoding="utf-8")
        todos = json.load(arq)
        arq.close()

    if RODAR_CDC:
        X, y = ler_cdc()
        todos["cdc"] = rodar_base("cdc", "CDC Diabetes", X, y, ["Sem diabetes", "Com diabetes"],
                                  ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age"], 3, 10000, 30000, 20000)

    if RODAR_ESTUDANTES:
        X, y = ler_estudantes()
        todos["estudantes"] = rodar_base("estudantes", "Evasão de estudantes", X, y, ["Não evasão", "Evasão"],
                                       ["Age at enrollment", "Curricular units 1st sem (grade)",
                                        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (grade)",
                                        "Unemployment rate"], 5, None, None, None)

    ordem = {}
    for t in ["cdc", "estudantes"]:
        if t in todos:
            ordem[t] = todos[t]
    grafico_painel(ordem)

    arq = open("resultados/resultados.json", "w", encoding="utf-8")
    json.dump(ordem, arq, indent=2, ensure_ascii=False)
    arq.close()
    print("\nfiguras e tabelas salvas na pasta resultados/")
