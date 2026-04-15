# DENGUE PREDICTION

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Run project

1) creat conda env and activate
```
conda creat --prefix ./env python=3.10
conda activate ./env
``` 
2) get in the project dir
``` 
cd kedro_project_name

``` 
## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter is already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)

# Data


## Data source 
**see notebooks/data_vizualisation for more info**

### Casos de arbovirose

[Portal de Dados Abertos do SUS](https://dadosabertos.saude.gov.br/dataset/arboviroses-dengue?utm)

[Conjunto de dados - Portal de Dados Abertos da Prefeitura do Recife](https://dados.recife.pe.gov.br/pt_BR/dataset/?tags=arbovirose)

### Dados meteorológicos

[INMET :: BDMEP](https://bdmep.inmet.gov.br/)