## 2023
#### Artificial Intelligence Approach for Severe Dengue Early Warning System 

###### Modelo :
**Modelos usados**
- Machine Learning (vários algoritmos)
- Deep Learning (LSTM)

**Previsão de surtos**
- Melhor modelo: **Extra Trees Classifier**
**Previsão de casos**
- Melhor modelo: **CatBoost Regressor**
###### Dados utilizados: 
- Dados de **2014 a 2021**
- 16 distritos + cidade
- Variáveis:
    - Epidemiológicas (casos de dengue)
    - Meteorológicas e climatológicas
- Total: **7208 amostras semanais**
###### **Limitações**
- Dados climáticos apenas no nível da cidade (não por distrito)
- Deep Learning (LSTM) teve desempenho inferior (dados insuficientes)
- Necessidade de mais granularidade (ex: IoT, sensores locais)

## 2024
###### Problemas:
- Muitos falam de falta da integração de múltiplas fontes (ex: comportamento humano, mobilidade, internet, população)
- presença de **relações não lineares**
- overfitting
- modelos conseguem seguir tendência e sazonalidade, mas ainda têm dificuldade em antecipar **explosões abruptas** de casos.
- Quase todos os artigos foram construídos e validados em um contexto geográfico específico
- **dependência de dados de qualidade**. Os trabalhos dependem de séries históricas consistentes, dados climáticos sem muitas falhas, registros epidemiológicos bem preenchidos e, em alguns casos, imputação de valores ausentes
###### comum:
- Dados climaticos
- Descoberta: sazonalidade
- **variáveis climáticas ajudam bastante**, especialmente quando se consideram **efeitos atrasados**
#### Dengue Early Warning System and Outbreak Prediction Tool in Bangladesh Using Interpretable Tree‐BasedMachine Learning Model
###### Modelo :
**Modelos usados**
- modelos de ML (Random Forest, XGBoost, LightGBM).
- Melhor modelo: **LightGBM**.
###### Dados utilizados: 
- Dados usados: **climáticos + sociodemográficos + ambientais + epidemiológicos** (2000–2021).
- Descoberta → Dengue depende fortemente de **temperatura, umidade, chuva e densidade populacional**.
######  **Diferencial:**
    - Uso de **SHAP** → modelo interpretável.
######  **Limitações:**
- **Dependência de dados locais:** Modelo treinado só em Bangladesh → baixa generalização.
- Dados agregados: Não considera granularidade fina (ex: bairros, microclima).
- **Foco em correlação:** Não modela causalidade epidemiológica real.


#### Application of multiple linear regression model and long short-term memory with compartmental model to forecast dengue cases in Selangor  Malaysia based on climate variables
*(Potencial para testar no Brasil)*
###### Modelo :
modelo híbrido combinando:
- Regressão linear (MLR)
- Deep Learning (LSTM)
- Modelo epidemiológico (SI-SIR)
###### Dados utilizados: 
- Dados usados: clima (temperatura, umidade, chuva)
######  **Diferencial:**
    Consegue prever até **60 semanas à frente**
######  **Limitações:**
- Modelo complexo  difícil de implementar e modelos são “caixa-preta”
- Depende de **muitos dados (clima + epidemiologia)**
- Relação clima → dengue ainda é **fraca/difícil de modelar diretamente**
#### development_of_data_driven_machine_learning_models 
(Explorar nao linearidade dos dados e contexto local)
###### Modelo :
modelo usados:
- Regressão (MLR, polinomial)
- Árvores de decisão
- Random Forest
- SVM
- Redes neurais (ANN)
- Séries temporais (ARIMA, SARIMA, Prophet)
###### Dados utilizados: 
- Temperatura, Umidade, Chuva e Vento + Casos históricos de dengue
- Descoberta: Relação é **não linear e depende do contexto local**.
######  **Diferencial:**
	Análise (correlação, gráficos)
######  **Limitações:**
- Falta integração de múltiplas fontes (ex: comportamento humano)
#### Leveraging Climate Data for Dengue Forecasting in Ba Ria Vung Tau Province, Vietnam An Advanced Machine Learning Approach
###### Modelo :
modelo usados:
- NBR
- SARIMAX
- XGBoost
- LSTM
Melhor modelo: **Negative Binomial Regression (NBR)**
Descoberta: Modelos conseguem capturar  *sazonalidade e tendências*
###### Dados utilizados: 
- 2003–2022 (casos + clima)
- Temperatura, umidade, chuva, vento, pressão, etc.
- Feature engineering (lags, médias móveis, sazonalidade)
- Descoberta: Uso de **lags climáticos** melhorou bastante a previsão
######  **Diferencial:**
	Análise (correlação, gráficos)
######  **Limitações:**
- modelos ainda precisam melhorar
- dependem fortemente de dados climáticos

#### Precision Prediction for Dengue Fever in Singapore A Machine Learning Approach Incorporating Meteorological Data
###### Modelo :
modelo usados:
- GLM
- SVM 
- Decision Tree 
- RF
- GBM 
- **XGBoost**
Melhor modelo: **XGBoost**
Descoberta: Modelos conseguem capturar  *sazonalidade e tendências*
###### Dados utilizados: 
- 2012–2022 (casos semanais + clima)
- Temperatura, chuva, vento, radiação solar, UV, nuvens
- lags (1–12 semanas)
- Descoberta:  Variáveis importantes → tempo (semana), cobertura de nuvens, chuva (com atraso), ponto de orvalho
######  **Diferencial:**

######  **Limitações:**
- modelos ainda precisam melhorar
- dependem fortemente de dados climáticos
- falhar em surtos inesperados (mudanças abruptas)

#### When climate variables improve the dengue forecasting a machine learning approach
###### Modelo :
modelo usados: Random Forest
Descoberta:  Modelos precisam ser adaptados localmente
###### Dados utilizados: 
- diferentes combinações de dados  → variáveis climáticas realmente melhoram a previsão ?
	-  **D** → só dengue
	- **CD** → dengue + clima
	- **HD** → dengue + umidade
- Cidades: Brasil (Natal), Peru (Iquitos), Colômbia (Barranquilla)
- Descoberta:  Não existe solução única
	- 🇧🇷 Natal → melhor usar **só dengue (D)**
	- 🇵🇪 Iquitos → melhor usar **clima + dengue (CD)**
	- 🇨🇴 Barranquilla → melhor usar **umidade + dengue (HD)**

	Variável mais importante → umidade
######  **Diferencial:**

######  **Limitações:**

#### !!(Diferentes tipos de dados)!! A reproducible ensemble machine learning approach
Previsão da taxa de incidência de dengue (DIR) com 1 mês de antecedênci
###### Modelo :
modelo usados: 
	Ensemble de Machine Learning combinando:
		- CatBoost (gradient boosting)
		- SVM (Support Vector Machine)
		- LSTM (rede neural recorrente)
	Saída final combinada via Random Forest
Descoberta:  
###### Dados utilizados: 
- Casos de dengue (2001–2019) por estado (SINAN)
- Tipos de dados: 
	- Dados populacionais (IBGE)
	- Variáveis climáticas: Temperatura, precipitação, umidade, vento
	- Variáveis ambientais: NDVI (vegetação), altitude, perda florestal
	- Variáveis socioeconômicas (31 indicadores)
	- Dados satelitais (ERA5, MODIS, Landsat)
- Descoberta:  
	- Integração de dados multimodais melhora significativamente o desempenho
######  **Diferencial:**
- Tipos de de dados variados
- Capacidade de generalização: modelo treinado no Brasil funcionou também no Peru
######  **Limitações:**
- Dificuldade em prever valores extremos (surtos muito abruptos)
- Dependência de qualidade e disponibilidade de dados
## 2025
###### Problemas:
- Não considera fatores sociais (mobilidade, densidade populacional)
- Falta de variáveis socioeconômicas e sociais: mobilidade, condições socioeconômicas, saneamento, comportamento populacional, desigualdade territorial
- Alto custo computacional dos modelos mais fortes : LSTM, modelos híbridos, SHAP, INLA, pipelines multi-etapas
###### Comum:
-  variáveis: clima, tempo, espaço, ambiente, população
-  dependência temporal com atraso,  efeito climático impactam os casos semanas ou meses depois (**efeito de lag**)
- Modelos híbridos ou mais ricos superam modelos isolados
- Modelos híbridos ou mais ricos superam modelos isolados
#### Assessing dengue forecasting methods a comparative study of statistical models and machine learning techniques in Rio de Janeiro
###### Modelo :
modelo usados: 
- Estatísticos: AR, MA, ARIMA, ETS, VAR, SARIMAX
- Machine Learning: Random Forest, XGBoost, SVM, LSTM, Prophet
- Ensemble: combinação de modelos (ex: LSTM + ARIMA)
Descoberta:  
- Modelos com variáveis climáticas (temperatura e umidade) tiveram melhor desempenho geral
- LSTM foi o modelo mais preciso no geral
- Prophet teve melhor desempenho em previsões de longo prazo
- ARIMA foi o melhor entre os modelos puramente estatísticos
- Modelos ensemble superaram modelos individuais
###### Dados utilizados: 
- Casos semanais de dengue no Rio de Janeiro (2016–2023)
- temperatura e umidade
######  **Diferencial:**
- dados semanais → **janela móvel (rolling window)** 
######  **Limitações:**
- Considera apenas variáveis climáticas (não inclui fatores sociais, mobilidade, densidade populacional)
-  LSTM, apesar de mais preciso, tem maior custo computacional
- Janela de treino (6 anos) não foi otimizada profundamente

#### Comparison of Deep Learning and Gradient Boosting  ANN Versus XGBoost for Climate‐Based DenguePrediction in Bangladesh
###### Modelo :
modelo usados: 
- Deep Learning: ANN (Artificial Neural Network)
- Machine Learning: XGBoost
- Baseline: Regressão Linear
melhor modelo: XGBoost
Descoberta:  
	Modelos baseados em árvore funcionam melhor com dados climáticos
###### Dados utilizados: 
- Casos mensais de dengue em Bangladesh (2000–2023)
- Temperatura (média, mínima, máxima), Umidade relativa, Precipitação, Pressão, Velocidade do vento
- Descoberta:  
	-  Precipitação é o fator mais importante para dengue
	- Umidade relativa e velocidade do vento também influenciam fortemente
	- Temperatura foi removida do modelo final por multicolinearidade e baixa relevância
######  **Diferencial:**
- Uso de **dados de longo prazo (23 anos)**
- Tratamento explícito de **multicolinearidade (VIF)**
######  **Limitações:**
	Dados mensais (menos granular)
#### Forecasting dengue across Brazil with LSTM neural networks and SHAP-driven lagged climate and spatial effects
###### Modelo :
modelo usados: 
- LSTM (Long Short-Term Memory)
- SHAP (Shapley Additive Explanations)
Descoberta:  
- LSTM + clima + efeito espacial foi o melhor modelo
- SHAP melhorou a seleção de variáveis → maior precisão
###### Dados utilizados: 
- Casos semanais de dengue em todos os 27 estados do Brasil
- Temperatura, Umidade, Precipitação, Pressão atmosférica, Dias de chuva
- Variáveis climáticas com lag (1–3 meses)
- Descoberta:  
	- Inclusão de dados de estados vizinhos aumentou performance, existe dependência espacial (regiões vizinhas influenciam surtos)
	- Modelos que combinam **tempo + clima + espaço + ambiental** são superiores
######  **Diferencial:**
- Espacial (vizinhança geográfica)
- Uso de:
	- Janela móvel (7 anos)
	- Previsão de médio prazo (1 e 3 meses)
######  **Limitações:**
-  Modelo mais complexo e computacionalmente caro
- Não considera fatores socioeconômicos diretamente

#### Integrating meteorological data and hybrid intelligent models for dengue fever prediction
###### Modelo :
modelo usados: 
- Base estatística: DLNM (Distributed Lag Nonlinear Model)
- Machine Learning: SVM, Random Forest (RF), KNN
- Otimização: IHLOA (Improved Horned Lizard Optimization Algorithm)
- Clustering: Fuzzy clustering (CV-WPFCM)
Descoberta:  
- Modelos híbridos (multi-etapas) tiveram melhor desempenho
- Feature selection com IHLOA melhorou muito a acurácia
- Considerar lag não linear (DLNM) é essencial
###### Dados utilizados: 
- Casos mensais de dengue (2005–2024)
- Regiões: Guangdong e Zhejiang (China)
- Temperatura, Umidade, Pressão, Vento, Visibilidade, NDVI (vegetação), Duração do sol
- Descoberta:  
	- Correlação positiva: temperatura, umidade, sol, vegetação
	- Correlação negativa: vento, pressão, visibilidade
	- Existe forte **efeito de lag não linear** entre clima e dengue
######  **Diferencial:** pipeline completo
1) Clustering → classifica risco (baixo/médio/alto)
2) DLNM → captura lag não linear
3) Feature selection → IHLOA (metaheurística avançada)
4) Predição → SVM / RF / KNN
######  **Limitações:**
- Alto custo computacional
- Muitos hiperparâmetros e etapas → risco de overfitting
- Dados mensais (menos detalhado que semanal)
- Interpretabilidade mais difícil (muitas camadas)

#### Predicting spatio-temporal dynamics of dengue using INLA (integrated nested laplace approximation) in Yogyakarta, Indonesia
###### Modelo :
modelo usados (modelagem estatística):  Modelo Bayesiano espaço-temporal (INLA – Integrated Nested Laplace Approximation)
Descoberta:  
###### Dados utilizados: 
- Granularidade: 78 subdistritos / dados mensais
- Região: Yogyakarta (Indonésia)
- Período: 2017–2022
- Tipos de dados:
	-  Epidemiológicos (casos de dengue)
	- Climáticos: chuva (com lag), temperatura, umidade, vento, pressão
	- Sociodemográficos: densidade populacional
	- Ambientais: uso do solo (urbano, vegetação, água, etc.)
- Descoberta:  
-  Variáveis mais importantes: chuva com lag (1–2 meses), temperatura, umidade e áreas urbanas e água
- Dengue depende de múltiplos fatores combinados
######  **Diferencial:**
- Uso de **modelo Bayesiano (não ML tradicional)**
- Uso de:
	- dados geográficos detalhados (subdistrito)
	- sensoriamento remoto (uso do solo)
######  **Limitações:**
 - Modelo mais complexo matematicamente (difícil de implementar)
 - Requer conhecimento em:
	- estatística Bayesiana
	- modelagem espacial

## Reflexao
**fatores sociais** ainda entram menos do que clima/ambiente, e **previsão de surtos bruscos/extremos** também parece menos explorada do que previsão “média” de séries temporais.
### Artigos que elaboram essa lacuna (2025)
####  (tipos de dados) Dengue dynamics beyond biological factors Revealing the nexus between urbanisation planning and mobilities in Vientiane Lao PDR
###### Modelo :
modelo usados: Regressão Binomial Negativa (modelo estatístico
###### Dados utilizados: 
- Casos de dengue (2012–2018) geolocalizados por vila
- Tipos de dados:
	- Censo populacional (demografia, migração, condições de moradia)
	- Dados urbanos: Expansão de áreas construídas (GHSL), Infraestrutura (OpenStreetMap)
	- Dados de mobilidade: Meta/Facebook (fluxo população dia/noite)
	- Dados ambientais: Temperatura (Landsat)
	- Dados socioeconômicos (acesso à água, renda, etc.)
- Descoberta:  Fatores urbanos são **mais consistentes que fatores biológicos isolados**
	- Maior incidência de dengue em áreas **recentemente urbanizadas**
	- Alta mobilidade diária (fluxo dia/noite) aumenta o risco
	- Migração (pessoas de fora da cidade) está associada a maior incidência
	- Infraestrutura (acesso à água encanada) reduz o risco
######  **Diferencial:**
- Integração forte entre:
	- Urbanização
	- Mobilidade humana
	- Infraestrutura urbana
######  **Limitações:**
- Não é um modelo preditivo (foco explicativo)
- Estudo restrito a uma cidade (Vientiane)
- Dependência de dados locais específicos
- Resultados podem não generalizar facilmente para outros contextos
#### (tipos de dados) Annual global dengue dynamics are related to multi-source factors revealed by a machine learning prediction analysis
###### Modelo :
modelo usados: Random Forest, XGBoost, MLP e SVR
Interpretação dos fatores com **SHAP**
melhor modelo: Random Forest multivariável
Descoberta:  
###### Dados utilizados: 
- Tipos de dados:
	- Casos anuais de dengue em escala global
	- Casos históricos de dengue
	- População
	- Clima
	- Transporte aéreo
	- Cobertura florestal
	- Anemia
	- Presença de vetores
	- Sorotipos virais
	- Indicadores socioeconômicos
- Descoberta:  
######  **Diferencial:**
- Escala **global**, e não apenas local ou nacional. 
- Foco em **previsão anual**
- Integração de **múltiplas fontes de dados** em um único modelo
######  **Limitações:**
- Casos históricos dominam fortemente a predição, o que pode reduzir o peso prático dos demais fatores
- Desempenho varia entre regiões
#### limate and environmental drivers of dengue expansion in São Paulo Brazil - An ecological niche modelling approach
O ENM (MaxEnt) faz mapa de risco (onde pode acontecer) 
Poder usar o ENM como **feature dentro de um modelo preditivo**.
###### Modelo :
modelo usados: Maximum Entropy (MaxEnt)
Descoberta:  
###### Dados utilizados: 
- Pegam municípios onde casos > percentil 95 (P95)
- temperatura (max e min), precipitação, altitude, NDVI (vegetação) e densidade populacional
- Descoberta:  densidade populacional
######  **Diferencial:**
- procura local propicio a surto de dengues
######  **Limitações:**
- Não inclui mobilidade
#### (tipos de dados) Dengue forecasting and outbreak detection in Brazil using LSTM -integrating human mobility and climate factors
###### Modelo :
modelo usados: LSTM 
Descoberta:  
###### Dados utilizados: 
- Dados de 10 cidades brasileiras entre 2016 e 2023
-  Tipo de dados:
	- Casos semanais de dengue por município
- Temperatura, Umidade
- População, Dados de mobilidade humana entre cidades
- Fluxos multimodais: rodoviário, hidroviário, aéreo
- Descoberta:  
######  **Diferencial:**
- Previsão **semanal** de casos
- Horizonte: **até 4 semanas (1 mês)**
- mobilidade humana
######  **Limitações:**
- Depende da disponibilidade e qualidade dos dados de mobilidade
- Não incorpora variáveis espaciais mais profundas, como uso do solo ou suitability ambiental
#### (casos anomalos) A statistical model for forecasting probabilistic epidemic bands for dengue cases in Brazil

Ele funciona como um **modelo de referência histórica + monitoramento de anomalia epidêmica**.
1. olha o **histórico de casos**
2. aprende o **comportamento esperado** por semana e por região
3. gera uma **distribuição provável** de casos futuros
4. transforma essa distribuição em **faixas epidêmicas**
5. compara o observado com essas faixas para dizer **quão anômala/atípica** está a epidemia
###### Modelo :
modelo usados: 
Descoberta:  
###### Dados utilizados: 
- Descoberta:  
######  **Diferencial:**
######  **Limitações:**

#### Machine Learning forecasting of dengue in São Paulo using virtual data augmentation and urban incident predictors - Addressing the exceptional surge of cases in 2024
###### Modelo :
modelo usados: 
- AdaBoost
- CatBoost (melhor desempenho)
- Random Forest
- XGBoost, LightGBM
- SVR, KNN
- Redes neurais (ANN)
- Modelos lineares (Ridge, Lasso, Elastic Net)
- Baselines: ARIMA, SARIMA, ETS
Descoberta:  
- ML consegue capturar relações não lineares e mudanças abruptas
- Modelos tradicionais falham quando os casos extrapolam o histórico
###### Dados utilizados: 
- Dados epidemiológicos (2014–2025):
- Dados climáticos: temperatura, precipitação, umidade
- Dados urbanos: alagamentos, ruas inundadas, deslizamentos, árvores caídas
- variáveis defasadas (lags: 2, 4, 8 semanas)
- Descoberta:  
######  **Diferencial:**
- Data augmentation virtual
- variáveis urbanas inéditas
- Foco explícito em **eventos extremos**
######  **Limitações:**
- Estudo focado apenas em São Paulo
- Alta complexidade do modelo (menos interpretável)
- Previsão de curto prazo (1 semana)