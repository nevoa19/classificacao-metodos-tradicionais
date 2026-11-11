import os

PASTA_TAB = "resultados/tabelas"

def numero_br(x, casas=3):
    if type(x) == str:
        return x
    if type(x) == int:
        return f"{x:,}".replace(",", ".")
    texto = ("%." + str(casas) + "f") % x
    return texto.replace(".", ",")

def salvar_tabela(nome, cabecalho, linhas, legenda, rotulo, alinhamento=None):
    if not os.path.exists(PASTA_TAB):
        os.makedirs(PASTA_TAB)
    if alinhamento is None:
        alinhamento = "l" + "c" * (len(cabecalho) - 1)

    texto = "\\begin{table}[!t]\n\\centering\n"
    texto = texto + "\\caption{" + legenda + "}\n"
    texto = texto + "\\label{" + rotulo + "}\n"
    texto = texto + "\\begin{tabular}{" + alinhamento + "}\n\\hline\n"
    i = 0
    while i < len(cabecalho):
        texto = texto + cabecalho[i]
        if i < len(cabecalho) - 1:
            texto = texto + " & "
        i += 1
    texto = texto + " \\\\\n\\hline\n"
    for linha in linhas:
        i = 0
        while i < len(linha):
            texto = texto + str(linha[i])
            if i < len(linha) - 1:
                texto = texto + " & "
            i += 1
        texto = texto + " \\\\\n"
    texto = texto + "\\hline\n\\end{tabular}\n\\end{table}"

    arq = open(PASTA_TAB + "/" + nome + ".tex", "w", encoding="utf-8")
    arq.write(texto)
    arq.close()
