from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RECIFE_CKAN_PACKAGE_URL = (
    "https://dados.recife.pe.gov.br/api/3/action/package_show"
    "?id=casos-de-dengue-zika-e-chikungunya"
)
INMET_YEAR_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
YEARS = range(2013, 2022)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw Recife dengue and INMET A301 weather data."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download files that already exist.",
    )
    args = parser.parse_args()

    dengue_dir = DATA_DIR / "source" / "dengue_data" / "Recife"
    weather_dir = DATA_DIR / "source" / "PE" / "weather_data"
    dengue_dir.mkdir(parents=True, exist_ok=True)
    weather_dir.mkdir(parents=True, exist_ok=True)

    dengue_resources = _recife_dengue_resource_urls()
    for year in YEARS:
        output_path = dengue_dir / f"casos-de-dengue-{year}.csv"
        if args.skip_existing and output_path.exists():
            print(f"[skip] {output_path}")
            continue

        print(f"[download] Recife dengue {year}")
        _download_file(dengue_resources[year], output_path)

    temp_dir = DATA_DIR / "source" / "_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for year in YEARS:
            output_path = weather_dir / (
                f"INMET_NE_PE_A301_RECIFE_01-01-{year}_A_31-12-{year}.csv"
            )
            if args.skip_existing and output_path.exists():
                print(f"[skip] {output_path}")
                continue

            zip_path = temp_dir / f"inmet_{year}.zip"
            print(f"[download] INMET historical data {year}")
            _download_file(INMET_YEAR_URL.format(year=year), zip_path)
            _extract_recife_weather_csv(zip_path, output_path, year)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nRaw data is ready under data/source.")


def _recife_dengue_resource_urls() -> dict[int, str]:
    with urlopen(RECIFE_CKAN_PACKAGE_URL, timeout=60) as response:
        import json

        payload = json.load(response)

    resources = payload["result"]["resources"]
    urls: dict[int, str] = {}
    for resource in resources:
        name = str(resource.get("name", ""))
        url = str(resource.get("url", ""))
        if "Dengue" not in name or not url.lower().endswith(".csv"):
            continue
        for year in YEARS:
            if str(year) in name:
                urls[year] = url

    missing = [year for year in YEARS if year not in urls]
    if missing:
        raise RuntimeError(f"Missing Recife dengue download URLs for years: {missing}")
    return urls


def _download_file(url: str, output_path: Path, retries: int = 3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 dengue_prediction data bootstrap "
                        "(educational research)"
                    )
                },
            )
            with urlopen(request, timeout=120) as response, open(temp_path, "wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            temp_path.replace(output_path)
            return
        except Exception as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            if attempt < retries:
                wait_seconds = 5 * attempt
                print(f"[retry] {url} failed ({exc}); waiting {wait_seconds}s")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to download {url}") from last_error


def _extract_recife_weather_csv(zip_path: Path, output_path: Path, year: int) -> None:
    expected_fragment = f"A301_RECIFE_01-01-{year}_A_31-12-{year}"
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            member
            for member in archive.namelist()
            if "A301_RECIFE" in Path(member).name
            and expected_fragment in Path(member).name
            and Path(member).suffix.lower() == ".csv"
        ]

        if not matches:
            raise RuntimeError(
                f"Could not find Recife A301 weather CSV for {year} inside {zip_path}."
            )

        with archive.open(matches[0]) as source, open(output_path, "wb") as target:
            shutil.copyfileobj(source, target)


if __name__ == "__main__":
    main()
