"""
CSE 163 Group Project
Luotong Kang, Yiyang Chen

A function file that loads and cleans datasets.
"""
import os
import pandas as pd
import geopandas as gpd
from dataprep.clean import clean_country

PATH = "/Users/luotongkang/文稿/Winter2022/CSE 163/Project/home"


def load_and_clean_22height():
    filename = "Height of Male and Female by Country 2022.csv"
    height_22 = pd.read_csv(os.path.join(PATH, filename))
    height_22 = clean_country(height_22, "Country Name",
                              output_format="alpha-3")
    return height_22


def load_and_clean_height_by_age():
    filename = "NCD_RisC_Lancet_2020_height_child_adolescent_country.csv"
    age_diff = pd.read_csv(os.path.join(PATH, filename))
    age_diff = clean_country(age_diff, "Country", output_format="alpha-3")
    age_diff = age_diff[["Country", "Country_clean", "Year", "Sex",
                         "Age group", "Mean height"]]
    duplicate_row = age_diff[age_diff["Country"] == "China (Hong Kong SAR)"]
    age_diff = age_diff.drop(duplicate_row.index)
    age_diff["Born Year"] = age_diff["Year"] - age_diff["Age group"]
    return age_diff


def load_and_clean_GDP():
    filename = "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3731360.csv"
    GDP = pd.read_csv(os.path.join(PATH, filename), skiprows=4)
    GDP = clean_country(GDP, "Country Name", output_format="alpha-3")
    GDP = GDP[GDP["Country Name_clean"].notnull()]
    GDP = GDP[(GDP["Country Code"] != "HKG") & (GDP["Country Code"] != "MAC")]
    return GDP


def load_and_clean_map():
    df = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    df["name_clean"] = clean_country(pd.DataFrame(df["name"]), "name",
                                     output_format="alpha-3")["name_clean"]
    df = df[df["name_clean"].notnull()]
    return df
