"""
CSE 163 Group Project
Luotong Kang, Yiyang Chen

A running script that countains visualization functions except for question 2
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from load_and_clean import load_and_clean_22height
from load_and_clean import load_and_clean_height_by_age
from load_and_clean import load_and_clean_GDP
from load_and_clean import load_and_clean_map
from plotly_regression_plot import melt_and_merge, regression_and_plot

sns.set()


def question1(countries, height_22):
    """
    Visualizes 2022 mean height for each country as a map.
    """
    merged_df_q1 = countries.merge(height_22, left_on="name_clean",
                                   right_on="Country Name_clean")
    fig, axs = plt.subplots(nrows=2)
    merged_df_q1.plot(column="Male Height in Cm", legend=True, ax=axs[0])
    merged_df_q1.plot(column="Female Height in Cm", legend=True, ax=axs[1])
    axs[0].set_title("Male height", fontsize=10)
    axs[1].set_title("Female height", fontsize=10)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle("World 2022 mean height in CM", fontsize=15, x=0.6, y=0.95)
    plt.gcf().set_size_inches(20, 8)
    plt.savefig('q1.png', bbox_inches='tight')


def question3_1(countries, GDP_19):
    """
    Visualizes 2019 GDP per capita for each country as a map.
    """
    GDP_19_by_countries = countries.merge(GDP_19, left_on='name_clean',
                                          right_on='Country Name_clean')
    fig, ax = plt.subplots(1, figsize=(15, 7))
    GDP_19_by_countries.plot(column="2019", legend=True, ax=ax)
    plt.title("2019 GDP per capita", fontsize=15, y=1.1)
    plt.savefig('q3_1.png', bbox_inches='tight')


def question3_2(age_18, GDP_19):
    """
    Visualizes relation between age 18 adolescent (boyes and girls) mean
        height and GDP per capita in 2019.
    """
    age18_year19 = age_18[age_18["Year"] == 2019]
    age18_change_by_GDP = age18_year19.merge(GDP_19, left_on='Country_clean',
                                             right_on='Country Name_clean')
    sns.lmplot(x="2019", y="Mean height", data=age18_change_by_GDP,
               hue="Sex", legend=False)
    plt.xlabel("GDP")
    plt.legend(bbox_to_anchor=(1, 1), loc="best",
               borderaxespad=0., fontsize="x-large")
    plt.title("Correlation between 18 year old adolescent mean height and "
              "GDP per capita in 2019", y=1.05, fontsize=20)
    plt.gcf().set_size_inches(20, 10)
    plt.savefig('q3_2.png', bbox_inches='tight')


def question3_3(age_18, GDP_19):
    """
    Visualizes age 18 adolescent (boyes and girls) mean height of
        each countries over time by 2019 GDP.
    """
    fig, axes = plt.subplots(2)
    age_18_by_discrete_19_GDP = age_18.merge(GDP_19[GDP_19["2019"] < 13000],
                                             left_on='Country_clean',
                                             right_on='Country Name_clean')
    age_18_by_discrete_19_GDP = age_18.merge(GDP_19[GDP_19["2019"] < 13000],
                                             left_on='Country_clean',
                                             right_on='Country Name_clean')
    girls_filter = age_18_by_discrete_19_GDP["Sex"] == "Girls"
    boys_filter = age_18_by_discrete_19_GDP["Sex"] == "Boys"
    sns.scatterplot(x="2019", y="Mean height",
                    data=age_18_by_discrete_19_GDP[girls_filter], hue="Year",
                    legend=False, ax=axes[0], size=1)
    sns.scatterplot(x="2019", y="Mean height",
                    data=age_18_by_discrete_19_GDP[boys_filter], hue="Year",
                    legend=False, ax=axes[1], size=1)
    axes[0].set_title("Girls", fontsize=10)
    axes[1].set_title("Boys", fontsize=10)
    axes[0].set_xlabel("2019 GDP")
    axes[1].set_xlabel("2019 GDP")
    plt.suptitle("18-year-old adolescent mean height over time vs. 2019 GDP "
                 "per capita (countries with GDP/capita<13000)",
                 y=0.95, fontsize=18)
    plt.gcf().set_size_inches(15, 10)
    plt.savefig('q3_3.png', bbox_inches='tight')


def preprocess_q4(age_diff, GDP_19):
    """
    Prepares the dataframe needed to visualize question 4.
    Returns a dataframe that contains complete track of 2 generation's
        mean height of two countries; and a dataframe that contains
        5 to 19 height difference of different born years in each country.
    """
    flt1 = (age_diff["Born Year"] >= 1980) & (age_diff["Born Year"] <= 2000)
    complete_tracks = age_diff[flt1]

    # for 4-1
    cn_filter = complete_tracks["Country"] == "China"
    us_filter = complete_tracks["Country_clean"] == "USA"
    cn_and_us = complete_tracks[cn_filter | us_filter]
    first_gen_filter = cn_and_us["Born Year"] == 1980
    second_gen_filter = cn_and_us["Born Year"] == 2000
    cn_and_us_2_gen = cn_and_us[first_gen_filter | second_gen_filter]

    # for 4-2
    age5_filter = complete_tracks["Age group"] == 5
    age19_filter = complete_tracks["Age group"] == 19
    flt2 = (age5_filter | age19_filter) & (complete_tracks["Sex"] == "Boys")
    growth = complete_tracks[flt2]
    temp = pd.DataFrame(growth[["Country_clean", "Born Year", "Mean height"]])
    temp_str = pd.Series(growth["Age group"].astype(str))
    temp["Age group"] = temp_str
    growth_boys = temp.pivot(index=["Country_clean", "Born Year"],
                             columns="Age group")["Mean height"].reset_index()
    growth_boys["Growth"] = growth_boys["19"] - growth_boys["5"]
    growth_boys_by_GDP_19 = growth_boys.merge(GDP_19, left_on='Country_clean',
                                              right_on='Country Name_clean')

    return cn_and_us_2_gen, growth_boys_by_GDP_19


def question4_1(cn_and_us_2_gen):
    """
    Visualizes height tracking of 2 generation (born 1980/2000) of CHN and USA
    """
    rel = sns.relplot(x="Age group", y="Mean height", data=cn_and_us_2_gen,
                      hue="Country", style="Born Year", col="Sex",
                      kind="line", legend=True)
    suptitle = ("Tracking of mean height change of two generation"
                "in two countries")
    rel.fig.suptitle(suptitle, fontsize=20)
    rel.set_axis_labels("Age")
    rel._legend.set_bbox_to_anchor([0.95, 0.5])
    plt.gcf().set_size_inches(20, 10)
    plt.savefig('q4_1.png')


def question4_2(growth_boys_by_GDP_19):
    """
    Visualizes regression of mean growth by GDP per capita
    """
    fig, axes = plt.subplots(2)
    sns.regplot(x="2019", y="Growth", data=growth_boys_by_GDP_19,
                scatter_kws={'s': 12}, x_estimator=np.mean, ax=axes[0])
    sns.regplot(x="2019", y="Growth", data=growth_boys_by_GDP_19,
                scatter_kws={'s': 12}, x_bins=120, ax=axes[1])
    axes[0].set_title("confidence interval very small", fontsize=10)
    axes[1].set_title("use x_bins=120 to make it more discrete", fontsize=10)
    axes[0].set_xlabel("2019 GDP")
    axes[1].set_xlabel("2019 GDP")
    plt.suptitle("Mean Growth by GDP per capita", fontsize=15, y=0.95)
    plt.gcf().set_size_inches(15, 10)
    plt.savefig('q4_2.png', bbox_inches='tight')


def main():
    # load and clean
    height_22 = load_and_clean_22height()
    age_diff = load_and_clean_height_by_age()
    GDP = load_and_clean_GDP()
    countries = load_and_clean_map()

    # q1
    question1(countries, height_22)

    # q2 (new library)
    df = melt_and_merge(GDP, age_diff, 2009, 2020)
    regression_and_plot(df)

    # q3
    GDP_19 = GDP[["Country Name", "Country Name_clean",
                  "Country Code", "2019"]]
    age_18 = age_diff[age_diff["Age group"] == 18]
    question3_1(countries, GDP_19)
    question3_2(age_18, GDP_19)
    question3_3(age_18, GDP_19)

    # q4
    cn_and_us_2_gen, growth_boys_by_GDP_19 = preprocess_q4(age_diff, GDP_19)
    question4_1(cn_and_us_2_gen)
    question4_2(growth_boys_by_GDP_19)


if __name__ == '__main__':
    main()
