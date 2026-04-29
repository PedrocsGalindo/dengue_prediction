import datetime
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from dengue_prediction.settings import DATA_DIR


def get_recife_dengue_data(test_size= 0.3) -> tuple[pd.DataFrame, pd.Series]:
    if not (Path(DATA_DIR) / "processed" / "datasets" / "recife_dataset.csv").exists():
        dataset = preprocess__recife__day__data(DATA_DIR)
    else:
        dataset = pd.read_csv(Path(DATA_DIR) / "processed" / "datasets" / "recife_dataset.csv")
    
    X = dataset.drop(columns=['casos_dengue'])
    y = dataset['casos_dengue']
    return X, y


def preprocess__recife__day__data(data_dir: str):
    source_data_dir = data_dir / "source"
    dfs_years = list(range(2013, 2022))

    dengue_dir = source_data_dir / "dengue_data" / "Recife"
    dengue_df = pd.DataFrame()

    for year in dfs_years:
        path = dengue_dir / f"casos-de-dengue-{year}.csv"
        if year in [2015, 2019]:
            df = pd.read_csv(path, sep=",", low_memory=False)
        else:
            df = pd.read_csv(path, sep=";", low_memory=False)
        df = clear_cs_dengue_df(df, year)
        dengue_df = pd.concat([dengue_df, df], ignore_index=True)

    weather_dir = source_data_dir / "weather_data"
    weather_df = pd.DataFrame()

    for year in dfs_years:
        df = weather_dir / f"INMET_NE_PE_A301_RECIFE_01-01-{year}_A_31-12-{year}.csv"
        df = pd.read_csv(df, sep=";", encoding="latin1", skiprows=8, decimal=",")
        df = clear_weather_df(df, year)
        weather_df = pd.concat([weather_df, df], ignore_index=True)

    cases_per_day = dengue_df.groupby(dengue_df["data"]).size().reset_index(name="contagem")

    mean_30_days_arlyer = pd.DataFrame()
    days_mean = weather_df.groupby(weather_df["data"]).mean()

    for col in days_mean.columns:
        mean_30_days_arlyer[col] = days_mean[col].shift(1).rolling(window=30).mean()

    dataset = pd.merge(mean_30_days_arlyer, cases_per_day, on="data")
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=['hora'])
    dataset = dataset.rename(columns={
        "contagem": "casos_dengue",
        "precipitacao_total_horario_mm": "precipitacao_total_media_mm",
        "pressao_atmosferica_ao_nivel_da_estacao_horaria_mb": "pressao_atmosferica_media_mb",
        "pressao_atmosferica_max_na_hora_ant_aut_mb": "pressao_atmosferica_max_media_mb",
        "pressao_atmosferica_min_na_hora_ant_aut_mb": "pressao_atmosferica_min_media_mb",
        "temperatura_do_ar_bulbo_seco_horaria_c": "temperatura_ar_media_c",
        "temperatura_do_ponto_de_orvalho_c": "temperatura_ponto_orvalho_media_c",
        "temperatura_maxima_na_hora_ant_aut_c": "temperatura_max_media_c",
        "temperatura_minima_na_hora_ant_aut_c": "temperatura_min_media_c",
        "temperatura_orvalho_max_na_hora_ant_aut_c": "temperatura_orvalho_max_media_c",
        "temperatura_orvalho_min_na_hora_ant_aut_c": "temperatura_orvalho_min_media_c",
        "umidade_rel_max_na_hora_ant_aut": "umidade_rel_max_media",
        "umidade_rel_min_na_hora_ant_aut": "umidade_rel_min_media",
        "umidade_relativa_do_ar_horaria": "umidade_relativa_media",        
        })
    
    dataset = dataset.sort_values("data").reset_index(drop=True)

    dataset["ano"] = dataset["data"].dt.year
    dataset["mes"] = dataset["data"].dt.month
    dataset["dia"] = dataset["data"].dt.day
    dataset["dia_da_semana"] = dataset["data"].dt.dayofweek
    dataset = dataset.drop(columns=["data"])
    
    output_path = data_dir / "processed" / "datasets" / "recife_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    return dataset

def normalize_column_name(col):
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col


def clear_cs_dengue_df(df, year):
    df.columns = [normalize_column_name(col) for col in df.columns]

    if year < 2021:
        df = df.rename(
            columns={
                "nu_notificacao": "id",
                "dt_notificacao": "data",
            }
        )
    else:
        df = df.rename(
            columns={
                "nu_notific": "id",
                "dt_notific": "data",
            }
        )
    df = df[["id", "data"]]

    df = df.dropna(subset=["data"])

    if year < 2015:
        df["data"] = pd.to_datetime(
            df["data"],
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce",
        )

        if df["data"].dt.time.unique() == [datetime.time(0, 0)]:
            df["data"] = pd.to_datetime(
                df["data"],
                format="%Y/%m/%d",
                errors="coerce",
            )
    elif year < 2021 or year in [2022, 2023]:
        df["data"] = pd.to_datetime(
            df["data"],
            format="%Y-%m-%d",
            errors="coerce",
        )
    else:
        df["data"] = pd.to_datetime(
            df["data"],
            format="%d/%m/%Y",
            errors="coerce",
        )

    df = df.drop_duplicates(["id"])

    return df


def clear_weather_df(df, year):
    df.columns = [normalize_column_name(col) for col in df.columns]

    for col in [
        "vento_rajada_maxima_m_s",
        "vento_direcao_horaria_gr_gr",
        "vento_velocidade_horaria_m_s",
    ]:
        if col in df.columns:
            df = df.drop(columns=col)

    df = df.rename(
        columns={
            "data_yyyy_mm_dd": "data",
            "hora_utc": "hora",
        }
    )
    df = df.dropna(subset=["data"])

    if year < 2019:
        df["hora"] = df["hora"].str.split(":").str[0].astype(int)
        df["data"] = pd.to_datetime(
            df["data"],
            format="%Y-%m-%d",
            errors="coerce",
        )
    else:
        df["hora"] = (
            df.iloc[:, 1].astype(str).str.extract(r"(\d{2})\d{2}\s*UTC")[0].astype("Int64")
        )
        df["data"] = pd.to_datetime(
            df["data"],
            format="%Y/%m/%d",
            errors="coerce",
        )

    df = df.replace(-9999, np.nan)
    df = df.loc[:, df.isnull().mean() < 0.3]

    cols = [col for col in df.columns if col not in ["data", "hora"]]
    df[cols] = df[cols].fillna(df[cols].rolling(window=6, min_periods=2).median())

    df = df.loc[df.isnull().mean(axis=1) < 0.6]
    return df
