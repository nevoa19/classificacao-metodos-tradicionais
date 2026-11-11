import math
import numpy as np
from dados import SEMENTE

N_BOOTSTRAP = 1000

def calcular_metricas(y_real, y_prev):
    y_real = list(y_real)
    y_prev = list(y_prev)
    vp = 0
    vn = 0
    fp = 0
    fn = 0
    i = 0
    while i < len(y_real):
        if y_real[i] == 1 and y_prev[i] == 1:
            vp += 1
        elif y_real[i] == 0 and y_prev[i] == 0:
            vn += 1
        elif y_real[i] == 0 and y_prev[i] == 1:
            fp += 1
        else:
            fn += 1
        i += 1

    acuracia = (vp + vn) / len(y_real)
    precisao = 0.0
    if vp + fp > 0:
        precisao = vp / (vp + fp)
    revocacao = 0.0
    if vp + fn > 0:
        revocacao = vp / (vp + fn)
    f1 = 0.0
    if precisao + revocacao > 0:
        f1 = 2 * precisao * revocacao / (precisao + revocacao)

    prec0 = 0.0
    if vn + fn > 0:
        prec0 = vn / (vn + fn)
    rev0 = 0.0
    if vn + fp > 0:
        rev0 = vn / (vn + fp)
    f1_0 = 0.0
    if prec0 + rev0 > 0:
        f1_0 = 2 * prec0 * rev0 / (prec0 + rev0)

    m = {}
    m["acuracia"] = acuracia
    m["precisao"] = precisao
    m["revocacao"] = revocacao
    m["f1"] = f1
    m["f1_macro"] = (f1 + f1_0) / 2
    m["confusao"] = [[vn, fp], [fn, vp]]
    return m

def intervalo_f1(y_real, y_prev):
    y_real = np.array(y_real)
    y_prev = np.array(y_prev)
    n = len(y_real)
    gerador = np.random.default_rng(SEMENTE)
    f1s = []
    b = 0
    while b < N_BOOTSTRAP:
        idx = gerador.integers(0, n, n)
        r = y_real[idx]
        p = y_prev[idx]
        vp = np.sum((r == 1) & (p == 1))
        fp = np.sum((r == 0) & (p == 1))
        fn = np.sum((r == 1) & (p == 0))
        if 2 * vp + fp + fn > 0:
            f1s.append(2 * vp / (2 * vp + fp + fn))
        else:
            f1s.append(0)
        b += 1
    return float(np.quantile(f1s, 0.025)), float(np.quantile(f1s, 0.975))

def mcnemar(y_real, prev_a, prev_b):
    y_real = list(y_real)
    prev_a = list(prev_a)
    prev_b = list(prev_b)
    so_a_acerta = 0
    so_b_acerta = 0
    i = 0
    while i < len(y_real):
        a_acertou = prev_a[i] == y_real[i]
        b_acertou = prev_b[i] == y_real[i]
        if a_acertou == True and b_acertou == False:
            so_a_acerta += 1
        if a_acertou == False and b_acertou == True:
            so_b_acerta += 1
        i += 1

    n = so_a_acerta + so_b_acerta
    if n == 0:
        return 1.0
    k = min(so_a_acerta, so_b_acerta)
    soma = 0
    j = 0
    while j <= k:
        soma = soma + math.comb(n, j)
        j += 1
    p = 2 * soma / (2 ** n)
    if p > 1:
        p = 1.0
    return float(p)
