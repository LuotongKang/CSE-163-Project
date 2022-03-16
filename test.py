"""
CSE 163 Group Project
Luotong Kang, Yiyang Chen

Tests important process of the project: cleaning and Q4 visualization
Copy functions from cse163_utils.py
Running this file with no error. Prints cleaning message, not error.
"""
import pandas as pd
import math
import numpy as np
from dataprep.clean import clean_country
from load_and_clean import (load_and_clean_height_by_age,
                            load_and_clean_GDP)
from run_script import preprocess_q4


def check_approx_equals(expected, received):
    """
    Checks received against expected, and returns whether or
    not they match (True if they do, False otherwise).
    If the argument is a float, will do an approximate check.
    If the arugment is a data structure will do an approximate check
    on all of its contents.
    """
    try:
        if type(expected) == dict:
            # first check that keys match, then check that the
            # values approximately match
            return expected.keys() == received.keys() and \
                all([check_approx_equals(expected[k], received[k])
                    for k in expected.keys()])
        elif type(expected) == list or type(expected) == set:
            # Checks both lists/sets contain the same values
            return len(expected) == len(received) and \
                all([check_approx_equals(v1, v2)
                    for v1, v2 in zip(expected, received)])
        elif type(expected) == float:
            return math.isclose(expected, received, abs_tol=0.001)
        else:
            return expected == received
    except Exception as e:
        print(f'EXCEPTION: Raised when checking check_approx_equals {e}')
        return False


def assert_equals(expected, received):
    """
    Checks received against expected, throws an AssertionError
    if they don't match. If the argument is a float, will do an approximate
    check. If the arugment is a data structure will do an approximate check
    on all of its contents.
    """
    assert check_approx_equals(expected, received), \
        f'Failed: Expected {expected}, but received {received}'


def test_cleaning():
    """
    Tests the cleaning of country names
    """
    df = pd.DataFrame(data={"country": ["US",
                                        "United States of America",
                                        "Republic of the Congo",
                                        "DR Congo"]})
    correct_name = ["USA", "USA", "COG", "COD"]
    cleaned_df = clean_country(df, "country", output_format="alpha-3")
    assert_equals(list(cleaned_df["country_clean"]), correct_name)


def test_mean_growth_confi_interval():
    """
    Proves the existence of confidence interval for Q4 mean growth over time
    """
    age_diff = load_and_clean_height_by_age()
    GDP = load_and_clean_GDP()
    GDP_19 = GDP[["Country Name", "Country Name_clean",
                  "Country Code", "2019"]]
    cn_and_us_2_gen, growth_boys_by_GDP_19 = preprocess_q4(age_diff, GDP_19)
    test_1 = growth_boys_by_GDP_19["Country_clean"] == "CHN"
    test_2 = growth_boys_by_GDP_19["Country_clean"] == "USA"
    max_test_1 = max(growth_boys_by_GDP_19[test_1]["Growth"])
    mean_test_1 = np.mean(growth_boys_by_GDP_19[test_1]["Growth"])
    max_test_2 = max(growth_boys_by_GDP_19[test_2]["Growth"])
    mean_test_2 = np.mean(growth_boys_by_GDP_19[test_2]["Growth"])
    assert_equals(max_test_1 == mean_test_1, False)
    assert_equals(max_test_2 == mean_test_2, False)


def main():
    test_cleaning()
    test_mean_growth_confi_interval()


if __name__ == '__main__':
    main()
