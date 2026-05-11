"""Populate R12 x SSP x (urban, rural) drinking-water access rates.

Sources
-------
.. TODO: confirm full citation for the projection files (likely WHO-UNICEF
   Joint Monitoring Programme as underlying data source; projection methodology
   TBD).

- File 1: ``data/water/demands/drinking_water_access/
  projections_people_merge_countries_10_25(in).csv`` — country x SSP
  x RCP x year, variable ``Improved water services``. No urban/rural split.
  Used here only for NAM and CHN imputation.
- File 2: ``data/water/demands/drinking_water_access/
  projections_people_UR_income_10_25.csv`` — country x SSP x RCP x
  year x (urban|rural) x income quintile, same variable. File 2 excludes
  NAM, CHN and PAO; it therefore only populates AFR, EEU, FSU, LAM, MEA,
  PAS, RCPA, SAS and WEU under the canonical R12 mapping in
  ``data/node/R12.yaml``.

Rules per R12 region
--------------------
1. File 2 is truth for the nine regions it covers:
       urban_rate[R] = sum(pop_imp_acc_ur_inc | tot_ur=urban, iso3 in R)
                     / sum(pop_ur_inc        | tot_ur=urban, iso3 in R)
       rural_rate analogous.

2. NAM and CHN: urban = rural = pop-weighted total ``share_acc`` from file 1
   across iso3 in the region. File 1 gives ~1.0 for both — a saturation
   ceiling, not a projection.

3. PAO: domain override, urban = rural = 0.99 for every SSP and year. File 1
   gives 0.25-0.32 for PAO which is a broken aggregate, not a bias offset,
   so it is not used.

Known regional offsets (file 1 biased low vs file 2, consistency-check result
across the nine overlap regions): EEU -3.4 pp, PAS -6.4 pp, WEU -8.3 pp. Not
corrected — recorded here as a boundary artifact of the two source files
rather than a defect of either.

Year coverage
-------------
File 2 years: 2020, 2025, ..., 2095. File 1 years: 2025, 2030, ..., 2095.
Target grid: 2010, 2020, 2030, ..., 2100, 2110. For target years below the
earliest source year we carry the earliest value back (2010 carries from
2020; 2010 and 2020 for NAM/CHN carry from 2025). For 2100 and 2110 we hold
the 2090 value — no padding with a legacy baseline row.

Output schema
-------------
Each CSV columns the existing basin set (BCU_name like ``2|AFR``) in the
legacy SSP2 file's order. The R12 rate is broadcast uniformly to every basin
in that region via the column suffix; the imputation rules operate at the
R12 level, so no basin-area downscaling is applied.
"""

import pandas as pd
import yaml

from message_ix_models.util import package_data_path

HARMONIZED = package_data_path("water", "demands", "harmonized", "R12")
DRINKING_WATER_ACCESS = package_data_path("water", "demands", "drinking_water_access")
NODE_YAML = package_data_path("node", "R12.yaml")

VARIABLE = "Improved water services"
SSPS = [1, 2, 3, 4, 5]
SSP_RCP = {1: "rcp26", 2: "rcp60", 3: "rcp60", 4: "rcp60", 5: "rcp60"}
TARGET_YEARS = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110]

FILE2_REGIONS = ["AFR", "EEU", "FSU", "LAM", "MEA", "PAS", "RCPA", "SAS", "WEU"]
FILE1_REGIONS = ["NAM", "CHN"]
PAO_OVERRIDE = {"urban": 0.99, "rural": 0.99}


def load_iso3_to_r12() -> dict[str, str]:
    """Map ISO-3 code to R12 region suffix (e.g. 'CAN' -> 'NAM')."""
    with NODE_YAML.open() as f:
        y = yaml.safe_load(f)
    mapping: dict[str, str] = {}
    for key, value in y.items():
        if not (isinstance(value, dict) and key.startswith("R12_")):
            continue
        region = key.replace("R12_", "")
        for iso in value.get("child", []):
            mapping[iso] = region
    return mapping


def load_file2() -> pd.DataFrame:
    df = pd.read_csv(DRINKING_WATER_ACCESS / "projections_people_UR_income_10_25.csv")
    return df[df["variable"] == VARIABLE].copy()


def load_file1() -> pd.DataFrame:
    df = pd.read_csv(
        DRINKING_WATER_ACCESS / "projections_people_merge_countries_10_25(in).csv"
    )
    return df[df["variable"] == VARIABLE].copy()


def file2_regional_rates(df: pd.DataFrame, iso2reg: dict[str, str]) -> pd.DataFrame:
    """Pop-weighted urban/rural rate per (SSP, year, R12) from file 2.

    Returns long frame: SSP, RCP, year, region, setting, rate.
    """
    df = df.assign(region=df["iso3"].map(iso2reg))
    df = df[df["region"].isin(FILE2_REGIONS)]
    agg = (
        df.groupby(["SSP", "RCP", "year", "region", "tot_ur"])
        .agg(
            num=("pop_imp_acc_ur_inc", "sum"),
            den=("pop_ur_inc", "sum"),
        )
        .reset_index()
    )
    agg["rate"] = (agg["num"] / agg["den"]).clip(0, 1)
    return agg.rename(columns={"tot_ur": "setting"})[
        ["SSP", "RCP", "year", "region", "setting", "rate"]
    ]


def file1_regional_rates(df: pd.DataFrame, iso2reg: dict[str, str]) -> pd.DataFrame:
    """Pop-weighted total rate per (SSP, year, R12) from file 1."""
    df = df.assign(region=df["iso3"].map(iso2reg))
    df = df[df["region"].isin(FILE1_REGIONS)]
    agg = (
        df.groupby(["SSP", "RCP", "year", "region"])
        .agg(num=("pop_acc", "sum"), den=("pop", "sum"))
        .reset_index()
    )
    agg["rate"] = (agg["num"] / agg["den"]).clip(0, 1)
    return agg[["SSP", "RCP", "year", "region", "rate"]]


def select_target_year_value(series: pd.Series, target: int) -> float:
    """Map target-grid year to a source-year value.

    - target below min available -> use earliest available
    - target above max available -> use latest target-relevant source year,
      which is 2090 (drop 2095 back to the last decadal point to match the
      decadal target grid)
    - otherwise exact match (all interior target years are decadal and
      present in source)
    """
    available = sorted(series.dropna().index)
    if not available:
        raise ValueError("no source data to select from")
    if target in series.index and pd.notna(series.get(target)):
        return float(series.loc[target])
    if target < available[0]:
        return float(series.loc[available[0]])
    # target > max: fall back to 2090 if present, else latest decadal <= max
    decadal_cap = max(y for y in available if y % 10 == 0)
    return float(series.loc[decadal_cap])


def region_rate_table(
    regional_long: pd.DataFrame,
    ssp: int,
    setting_col: str | None = None,
) -> pd.DataFrame:
    """Build target-year x region wide table for a given SSP.

    regional_long: long frame with SSP, RCP, year, region, rate (+ optional
    setting column).
    """
    ssp_str = f"SSP{ssp}"
    rcp = SSP_RCP[ssp]
    mask = (regional_long["SSP"] == ssp_str) & (regional_long["RCP"] == rcp)
    if setting_col is not None:
        mask &= regional_long["setting"] == setting_col
    f = regional_long[mask]
    wide = f.pivot(index="year", columns="region", values="rate").sort_index()
    out = pd.DataFrame(index=TARGET_YEARS, columns=wide.columns, dtype=float)
    for region in wide.columns:
        series = wide[region]
        for y in TARGET_YEARS:
            out.at[y, region] = select_target_year_value(series, y)
    return out


def legacy_column_order(setting: str) -> list[str]:
    """Column order from the existing SSP2 baseline (schema anchor)."""
    path = HARMONIZED / f"ssp2_regional_{setting}_connection_rate_baseline.csv"
    return list(pd.read_csv(path, index_col=0, nrows=0).columns)


def broadcast_region_to_basins(
    region_rates: pd.Series, columns: list[str]
) -> pd.Series:
    """For each column `<basin_id>|<REGION>`, pick region_rates[REGION]."""
    out = {}
    for col in columns:
        region = col.split("|")[-1]
        if region not in region_rates.index:
            raise KeyError(
                f"region {region!r} not in rate table {list(region_rates.index)}"
            )
        out[col] = region_rates[region]
    return pd.Series(out)


def build_ssp_setting_csv(
    ssp: int,
    setting: str,
    file2_long: pd.DataFrame,
    file1_long: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the basin-wide table for one (ssp, setting) output."""
    f2_wide = region_rate_table(file2_long, ssp, setting_col=setting)
    f1_wide = region_rate_table(file1_long, ssp, setting_col=None)

    columns = legacy_column_order(setting)
    rows = []
    for year in TARGET_YEARS:
        region_rates = {}
        for r in FILE2_REGIONS:
            region_rates[r] = f2_wide.at[year, r]
        for r in FILE1_REGIONS:
            region_rates[r] = f1_wide.at[year, r]
        region_rates["PAO"] = PAO_OVERRIDE[setting]
        series = broadcast_region_to_basins(pd.Series(region_rates), columns)
        series.name = year
        rows.append(series)
    out = pd.DataFrame(rows)
    out.index = pd.Index(TARGET_YEARS, name="")
    return out[columns]


def main() -> None:
    iso2reg = load_iso3_to_r12()

    f2 = load_file2()
    f1 = load_file1()
    file2_long = file2_regional_rates(f2, iso2reg)
    file1_long = file1_regional_rates(f1, iso2reg)

    for ssp in SSPS:
        for setting in ("urban", "rural"):
            table = build_ssp_setting_csv(ssp, setting, file2_long, file1_long)
            out = (
                HARMONIZED / f"ssp{ssp}_regional_{setting}_connection_rate_baseline.csv"
            )
            table.to_csv(out)
            print(
                f"ssp{ssp} {setting}: wrote {out.name} "
                f"(2020 mean={table.loc[2020].mean():.3f}, "
                f"2090 mean={table.loc[2090].mean():.3f})"
            )


if __name__ == "__main__":
    main()
