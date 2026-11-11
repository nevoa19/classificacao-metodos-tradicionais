# Classificação com métodos tradicionais

Mini-projeto 01 (IA): SVM (linear e RBF), regressão logística e k-NN em duas bases binárias.

- CDC Diabetes Health Indicators (OpenML 46841), alvo `Diabetes_binary`
- Predict Students' Dropout and Academic Success (Kaggle, `thedevastator/higher-education-predictors-of-student-retention`), alvo `Target` convertido em evasão (Dropout = 1) x não evasão (Graduate/Enrolled = 0)

## Requisitos
- Python 3.10 ou superior (testado com 3.13 e 3.14)
- numpy, pandas, scikit-learn, scipy, matplotlib

    pip install -r requirements.txt

## Como executar
    python classificacao.py

Roda as duas bases (cerca de 3 minutos). Para rodar só uma, alterar `RODAR_CDC` ou
`RODAR_ESTUDANTES` para `False` no início do arquivo.

## Arquivos
- `classificacao.py`: roda o experimento das duas bases
- `dados.py`: leitura das bases e subamostragem
- `modelos.py`: modelos, pipelines e grades de hiperparâmetros
- `metricas.py`: métricas, intervalo de confiança por bootstrap e teste de McNemar
- `graficos.py`: figuras
- `tabelas.py`: tabelas em LaTeX

## Dados
Os dois arquivos ficam em `data/`:
`cdc_diabetes.arff` (OpenML) e `dataset.csv` (Kaggle).

## Protocolo
- divisão treino/teste estratificada 80/20, semente 42
- GridSearchCV com validação cruzada estratificada só no treino (3 folds no CDC, 5 nos estudantes), critério F1 da classe positiva
- padronização dentro do Pipeline
- SVM e regressão logística com `class_weight="balanced"`
- CDC: SVM ajustada em subamostra de 10.000 exemplos; k do k-NN escolhido em 30.000 exemplos e modelo final com todo o treino
- classificador trivial (classe majoritária) como referência
- IC 95% do F1 por bootstrap (1.000 reamostragens) e teste de McNemar exato contra o melhor modelo
- fronteiras de decisão: modelos reajustados no plano das 2 primeiras componentes principais (ilustrativo)

## Saídas
Em `resultados/`:
`tabelas/` (tabelas .tex para o relatório), `figuras/` (pdf e png) e `resultados.json`.
Os valores também são impressos no terminal.
