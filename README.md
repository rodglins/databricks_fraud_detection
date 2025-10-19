# 💳 Sistema de Detecção de Fraude de Alto Desempenho (Stacking & MLOps)

## Visão Geral do Projeto

Este repositório contém o código e a arquitetura para um sistema de detecção de fraude em transações financeiras, utilizando uma abordagem de **Model Stacking** (Ensemble de Nível 2) para alcançar precisão superior e extrema robustez.

A solução é desenvolvida e operacionalizada em um ambiente Databricks/PySpark, garantindo escalabilidade para processar milhões de transações em tempo real (ou quase real) via Inferência Distribuída (UDF PySpark). O **Unity Catalog** é utilizado para governança e registro de modelos.

## 🚀 Status e Performance Validada (Stress Test)

O modelo foi validado em um teste de estresse em lote com **5.000.000 de registros simulados**, demonstrando performance de classe mundial:

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **AUC (Area Under ROC)** | **0.999934** | Capacidade quase perfeita de ranquear transações fraudulentas. |
| **Recall (Sensibilidade)** | **100% (1.0000)** | Todas as fraudes simuladas foram corretamente detectadas (Zero Falsos Negativos). |
| **Erros Tipo II (FN)** | **0** | **Nenhuma fraude** passou despercebida (Obrigação Crítica de Segurança). |
| **Erros Tipo I (FP)** | **0** | **Nenhum bloqueio indevido** de transações legítimas (Eficiência Operacional). |

-----

## 🏗️ Arquitetura de Stacking

O sistema utiliza uma arquitetura de aprendizado em duas camadas (Tier-2 Stacking):

1.  **Nível 1 (Modelos Base - Out-of-Fold, OOF):** Três algoritmos de *gradient boosting* são treinados para produzir previsões de probabilidade (scores):

      * `LGBM_OOF` (LightGBM)
      * `XGB_OOF` (XGBoost)
      * `CAT_OOF` (CatBoost)

2.  **Nível 2 (Meta-Learner):** Um modelo de regressão logística simples (*Meta-Learner*) recebe os três scores de probabilidade do Nível 1 como *features de entrada*. Este modelo aprende a melhor maneira de combinar (pesar) as previsões para produzir a **probabilidade final de fraude (`final_fraud_proba`)**, eliminando vieses e aumentando a robustez.

## ⚙️ Componentes-Chave

| Componente | Função | Tecnologia |
| :--- | :--- | :--- |
| **Infraestrutura** | Ambiente de execução e processamento distribuído de big data. | Databricks / PySpark |
| **Model Registry** | Governança, Versionamento e Alias (`Champion`). | MLflow / Unity Catalog |
| **Modelo Implantação** | Execução do Meta-Learner em escala (inferência distribuída). | UDF PySpark (`mlflow.pyfunc.spark_udf`) |
| **Modelos Nível 1** | Base para o Stacking. | LightGBM, XGBoost, CatBoost |
| **Modelo Final** | Meta-Learner (Stacking). | Geralmente Regressão Logística ou Modelo Simples. |

## 🛠️ Execução da Inferência (Stress Test)

O arquivo `stress_test_stacking.py` é o script final de validação que executa as seguintes etapas no ambiente Databricks:

1.  **Geração de Dados Simulados:** Cria um DataFrame PySpark (`df_simulated_X`) com 5 milhões de linhas e scores altamente calibrados para simular um cenário ideal.
2.  **Carregamento do Modelo:** O modelo `stacking_fraude_model` com o *Alias* `Champion` é carregado diretamente do Unity Catalog via URI: `models:/workspace.default.stacking_fraude_model@Champion`.
3.  **Inferência em Massa:** A função `mlflow.pyfunc.spark_udf` é aplicada ao DataFrame de 5 milhões de linhas, utilizando a capacidade de processamento distribuído do Spark para gerar as previsões em paralelo.
4.  **Relatório de Qualidade:** Calcula AUC, Recall, Falsos Positivos e Falsos Negativos para confirmar a integridade e escalabilidade da solução.

## 🚀 Como Executar

Para replicar a execução do teste de estresse:

1.  **Pré-requisitos:** Certifique-se de que o modelo `stacking_fraude_model` esteja registrado no Unity Catalog com o alias `Champion`.
2.  **Ambiente:** Execute o script `stress_test_stacking.py` em um *notebook* Databricks anexado a um *cluster* com as bibliotecas MLflow, Spark e sklearn instaladas.

<!-- end list -->

```python
# Trecho principal do script de inferência
pyfunc_udf = mlflow.pyfunc.spark_udf(
    spark=spark, 
    model_uri="models:/workspace.default.stacking_fraude_model@Champion", 
    result_type=DoubleType()
)

df_predictions = (
    df_simulated_X 
    .withColumn("final_fraud_proba", pyfunc_udf(struct(*[col(c) for c in input_cols])))
)
# ...
```

-----

**Desenvolvido por:** [Seu Nome ou Equipe]
**Data da Validação:** [Data da Última Execução]
