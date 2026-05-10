# DENGUE PREDICTION

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

This project is about Regression problem to predict the number of dengue cases in the day, using climate last days information. 

The dataset is segmented in date and location (municipio)

## Run project

```bash
python -m venv venv
venv\Scripts\activate 
# get in the project dir
cd dengue_prediction
``` 
## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

```
pip install -r requirements.txt
```

TPOT `1.1.0` still pulls `stopit`, which imports `pkg_resources`, so this project pins `setuptools<81` in `requirements.txt`.

## How to run your Kedro pipeline

```
kedro run
```

### Run a specific ML pipeline

```
kedro run --pipeline tpot
kedro run --pipeline h2o
kedro run --pipeline autosklearn
kedro run --pipeline linear_regression
```

On Windows, `autosklearn` is expected to run through Docker. `tpot` and `h2o` run directly in the project environment.

### Download raw Recife data

From the Kedro project directory (`dengue_prediction/`), run:

```powershell
python scripts/bootstrap_recife_data.py --skip-existing
```

This downloads Recife dengue CSVs for 2013-2021 from the Recife Open Data CKAN API and INMET annual historical ZIPs, extracting only the Recife automatic station `A301`.

## Linear regression baseline

The `linear_regression` pipeline is the first baseline for the regression problem of predicting daily dengue cases. It uses `data/processed/datasets/casos_de_dengue_dataset.csv`, removes the location identifiers (`uf` and `municipio`) from the model features, keeps `data` only for temporal splitting, and reports MAE, MSE, RMSE, and R2.

#### Reproduce the linear regression experiment

From the Kedro project directory, run:

```powershell
$env:PYTHONPATH = "src"
kedro run --pipeline linear_regression
```

The experiment uses the parameters in `conf/base/parameters.yml`:

```yaml
linear_regression:
  test_size: 0.2
  n_splits: 5
  fit_intercept: true
  positive: false
  scale_features: true
```

The protocol is:

- dataset: `data/processed/datasets/casos_de_dengue_dataset.csv`;
- target variable: `casos_dengue`, the daily number of dengue cases;
- removed columns: `uf` and `municipio`, because they identify the location rather than climate/time behavior;
- split-only column: `data`, reconstructed from `ano`, `mes`, and `dia`;
- features: weather variables plus calendar variables;
- train/test split: chronological holdout by unique date, with the final 20% of dates used as test data;
- validation: `TimeSeriesSplit` by unique date on the training data;
- leakage control: imputation and standardization are fitted only inside each training fold, and then only on the final training partition for the final test evaluation;
- metrics: MAE, MSE, RMSE, and R2.

Outputs are saved under:

- `dengue_prediction/data/results/linear_regression/<run_id>/report.json`
- `dengue_prediction/data/results/linear_regression/<run_id>/predictions.csv`
- `dengue_prediction/data/results/linear_regression/<run_id>/linear_regression.joblib`

To inspect the latest report in PowerShell:

```powershell
$report = Get-ChildItem data\results\linear_regression -Recurse -Filter report.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content -Raw $report.FullName | ConvertFrom-Json | Select-Object -ExpandProperty test_metrics
```

# Data

## Data source 
**see notebooks/data_vizualisation for more info**

### Casos de arbovirose

[Portal de Dados Abertos do SUS](https://dadosabertos.saude.gov.br/dataset/arboviroses-dengue?utm)

[Conjunto de dados - Portal de Dados Abertos da Prefeitura do Recife](https://dados.recife.pe.gov.br/pt_BR/dataset/?tags=arbovirose)

InfoDengue API ?? 

### Dados meteorológicos

[INMET :: BDMEP](https://bdmep.inmet.gov.br/)

### Utils
[Codigos para municipio e estados - IBGE](https://github.com/kelvins/municipios-brasileiros/blob/main/csv)
