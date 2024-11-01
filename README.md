
# **Inteligencia artificial**

A IA tem o objetivo de prever se uma pessoa é ou não um possível usuario
### **Criar uma base com pipeline para outras IAs**

Este notebook tem como o objetivo construir uma pipeline de machine learning para preparação de dados, treinamento e avaliação de modelos. A estrutura é projetada para que a base possa ser reutilizada e aplicada em diferentes inteligências artificiais com mínima adaptação.

## **Sumário**

1. [Introdução](#introdução)
2. [Conteúdo do Notebook](#conteúdo-do-notebook)
3. [Como Executar](#como-executar)
4. [Estrutura do Pipeline](#estrutura-do-pipeline)
5. [Modelos Implementados](#modelos-implementados)
6. [Serialização do Modelo](#serialização-do-modelo)
7. [Requisitos](#requisitos)

---

## Introdução

Este notebook implementa uma pipeline que prepara dados, trata desequilíbrios de classe e otimiza hiperparâmetros, proporcionando um fluxo de trabalho eficiente para análise preditiva. O pipeline inclui técnicas como validação cruzada, GridSearch e uso do SMOTE para balanceamento de dados.

---

## Conteúdo do Notebook

### 1. **Imports**
   - Importação de bibliotecas e pacotes para análise de dados e machine learning:
     - `Pandas` e `NumPy` para manipulação de dados.
     - `Scikit-learn` para algoritmos de machine learning, validação cruzada, e métricas.
     - `Imbalanced-learn` para técnicas de balanceamento, incluindo o **SMOTE**.
   - Exemplo:
     ```python
        from sklearn.model_selection import GridSearchCV, KFold, train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline 
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        import pandas as pd
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
        import numpy as np
        from sklearn import tree
        import matplotlib.pyplot as plt
        import joblib
     ```

### 2. **Conectando com a Base**
   - Carregamento dos dados de um arquivo, .xlsx, que os dados foram coletados do forms
   - Exemplo:
     ```python
     df = pd.read_excel('./base/incluses_tratado.xlsx')
     ```

### 3. **Separação entre Resposta e Atributos**
   - Separação das variáveis independentes e dependentes (X e y) para definir os dados de entrada e saída do modelo.
   - Exemplo:
     ```python
        df_resposta = df['Finalidade do Uso do App Incluses']
        df_atributo = df.iloc[:, :-1]
     ```

### 4. **Definindo o Modelo e a Pipeline incluindo o SMOTE**
   - Construção de uma pipeline com o SMOTE para balanceamento das classes, seguido pela definição do modelo.
   - Exemplo:
     ```python
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=1)),
            ('classifier', model)
        ])
     ```

### 5. **Definindo o Grid de Hiperparâmetros**
   - Configuração dos parâmetros a serem otimizados durante o GridSearch e os modelos.
   - Exemplo:
     ```python
        classifiers = {
        'knn': KNeighborsClassifier(),
        'naive_bayes': GaussianNB(),
        'decision_tree': DecisionTreeClassifier()
        }

        param_grid = [
            {  # Parâmetros para KNeighborsClassifier
                'classifier': [KNeighborsClassifier()],
                'classifier__n_neighbors': np.arange(1, 7, 1),
                'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            },
            {  # Parâmetros para GaussianNB
                'classifier': [GaussianNB()],
                'classifier__var_smoothing': np.logspace(0, 100, num=50), 
            },
            {  # Parâmetros para DecisionTreeClassifier
                'classifier': [DecisionTreeClassifier(random_state=12)],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__splitter': ['best', 'random'],
                'classifier__max_depth': [None, 2, 4, 6, 8, 10, 12],
                'classifier__min_samples_split': [2, 5, 10, None],
                'classifier__min_samples_leaf': [1, 2, 5, 10, None],
                'classifier__max_features': [None, 'sqrt', 'log2'],
            }

     ```

### 6. **Definindo Validação Cruzada e GridSearch**
   - Configuração da validação cruzada e execução do GridSearch para encontrar os melhores parâmetros.


### 7. **Testar Diferentes Divisões de Treino/Teste e Ajustar a Pipeline**
   - Avaliação do impacto de diferentes divisões entre treino e teste e ajuste da pipeline com base nos resultados.
   - Exemplo:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(df_atributo, df_resposta, test_size=0.25, random_state=42)
     ```

### 8. **Vendo os Melhores Parâmetros**
   - Identificação e aplicação dos melhores parâmetros encontrados nos modelos de machine learning escolhidos.
   - Modelos:
     - `DecisionTreeClassifier`
     - `KNeighborsClassifier`
     - `GaussianNB`

### 9. **Serialização do Modelo**
   - Salvamento do modelo treinado para uso posterior, facilitando a reutilização sem novo treinamento.
   - Exemplo:
     ```python
        pipeline = Pipeline(steps=[
            ('preprocessador', preprocessador),
            ('classificador_tree', classificador_tree)
        ])

        joblib.dump(pipeline, 'modelo_pipeline.pkl')
     ```

---

## Como Executar

1. Clone o repositório e acesse o diretório do notebook.
2. Instale os pacotes necessários, listados na seção [Requisitos](#requisitos).
3. Execute o notebook seguindo cada célula na ordem apresentada, ou use `Run All Cells`.
4. Use o modelo serializado (`modelo_pipeline.pkl`) para novas previsões ou análises.

---

## Estrutura do Pipeline

A pipeline é composta pelas etapas de:
1. **Pré-processamento com SMOTE** para balanceamento de classes.
2. **Modelo de machine learning** configurado com o melhor conjunto de hiperparâmetros encontrado pelo GridSearch.

---

## Modelos Implementados

- **DecisionTreeClassifier**: Modelo de árvore de decisão.
- **KNeighborsClassifier**: Classificador K-vizinhos mais próximos.
- **GaussianNB**: Algoritmo Naive Bayes Gaussiano.

---

## Serialização do Modelo

Após o treinamento, o modelo com os melhores parâmetros é salvo usando a biblioteca `joblib`. Isso permite que o modelo seja reutilizado facilmente em outros notebooks ou aplicações, sem a necessidade de reconfigurar ou retreinar.

---

## Requisitos

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib e Seaborn
- Joblib

## Feito por

[Luca Almeida Lucareli](https://github.com/LucaLucareli)

[Olivia Farias Domingues](https://github.com/oliviaworks)
