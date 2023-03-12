import numpy as np
import pandas as pd
import time
#%%

from thefuzz import fuzz
from thefuzz import process
#%%

from pandarallel import pandarallel
pandarallel.initialize(nb_workers = 8, progress_bar = False)
#%% read datasets

crsp = pd.read_csv(r"CRSPnames.csv")
se = pd.read_csv(r"SEmappingsDAFA.csv")
#%% drop duplicates from SEid column and convert DATE column into datetime format

se = se.drop_duplicates(subset = ["SEid"], keep = "first", ignore_index = True)
crsp["DATE"] = pd.to_datetime(crsp["DATE"], format = "%Y%m%d")
#%% filter companies based on the assumption that legal entities proxy for company's nationality

se_us = se[se["SECompanyName"].str.contains("|".join([" INC.", " INC", " Inc.", " Inc",
                                                   " LTD.", " LTD", " Ltd", " LP",
                                                   " PLC", " Plc", ".COM", ".Com"
                                                   " CO.", " CO", " Co.", " Co",
                                                   " CORP", " CORP.", " Corp", " Corp.",
                                                   " LLC", " Llc", " Lp", " Group",
                                                   " Trust", " plc"]))]
se_us = se_us.reset_index(drop = True)
#%% save companies assumed to be non american into separate dataframe

se_non_us = se[~se["SECompanyName"].str.contains("|".join([" INC.", " INC", " Inc.", " Inc",
                                                   " LTD.", " LTD", " Ltd", " LP",
                                                   " PLC", " Plc", ".COM", ".Com"
                                                   " CO.", " CO", " Co.", " Co",
                                                   " CORP", " CORP.", " Corp", " Corp.",
                                                   " LLC", " Llc", " Lp", " Group",
                                                   " Trust", " plc"]))]

se_non_us = se_non_us.reset_index(drop = True)

se_non_us["MergeComnam"] = "NON US COMPANY"
#%% define helper functions to create a small random df which is cleaned for testing

def rand_df(df_start):
    start = np.random.randint(0, len(df_start) - 1000)
    end = start + 1000
    df_rand = df_start.iloc[start:end].copy()
    
    return df_rand

def clean_df(df):
    df["SEHeadline_fix"] = df["SEHeadline"].str.split(" Earnings", 1).str[0].str.strip()
    df["SEHeadline_fix"] = df["SEHeadline_fix"].str.replace(r"\bQ[1-4]\b", " ")
    df["SEHeadline_fix"] = df["SEHeadline_fix"].str.replace(r"\b2[0-9]{3}\b", " ")
    df["SEHeadline_fix"] = df["SEHeadline_fix"].str.replace(r"[^a-zA-Z\d\s:]", " ")
    df["SEHeadline_fix"] = df["SEHeadline_fix"].str.replace("|".join([
                              "Half Year", "Full Year", "Interim", "Preliminary", "Interim",
                              "Conference", "First Quarter", "Second Quarter", "Third Quarter",
                              "Fourth Quarter", "Results ", "Call", "Fiscal", "FY"]), " ")
    df["SEHeadline_fix"] = df["SEHeadline_fix"].str.replace("Corporation", "Corp")
    df["SEHeadline_fix"] = df["SEHeadline_fix"].str.replace("Limited", "Ltd")
    
    return df

def prep_df(df_start):
    final_df = rand_df(df_start)
    final_df = clean_df(final_df)
    final_df = final_df.reset_index(drop = True)
    
    return final_df
#%% create testing dataframe for accuracy inference

df_test = prep_df(se_us)
#%% apply the fuzzy matching function to the testing dataframe

%%time
df_test["MergeComnam"] = df_test["SEHeadline_fix"].parallel_apply(lambda x: process.extractOne(x, list(crsp["COMNAM"]))[0])
#%% drop helper column

df_test = df_test.drop(columns = "SEHeadline_fix")
#%% take subset of total dataframe to run the same steps as above (step must be repeated until
# last row using non overlapping intervals, change iloc parameters to repeat the step)

se_us1 = se_us.iloc[10000:50000].copy()

se_us1["SEHeadline_fix"] = se_us1["SEHeadline"].str.split(" Earnings", 1).str[0].str.strip()
se_us1["SEHeadline_fix"] = se_us1["SEHeadline_fix"].str.replace(r"\bQ[1-4]\b", " ")
se_us1["SEHeadline_fix"] = se_us1["SEHeadline_fix"].str.replace(r"\b2[0-9]{3}\b", " ")
se_us1["SEHeadline_fix"] = se_us1["SEHeadline_fix"].str.replace(r"[^a-zA-Z\d\s:]", " ")
se_us1["SEHeadline_fix"] = se_us1["SEHeadline_fix"].str.replace("|".join([
                          "Half Year", "Full Year", "Interim", "Preliminary", "Interim",
                          "Conference", "First Quarter", "Second Quarter", "Third Quarter",
                          "Fourth Quarter", "Results ", "Call", "Fiscal", "FY"]), " ")
se_us1["SEHeadline_fix"] = se_us1["SEHeadline_fix"].str.replace("Corporation", "Corp")
se_us1["SEHeadline_fix"] = se_us1["SEHeadline_fix"].str.replace("Limited", "Ltd")

se_us1 = se_us1.reset_index(drop = True)
#%% apply the fuzzy matching function to the subset of dataframe

%%time
se_us1["MergeComnam"] = se_us1["SEHeadline_fix"].parallel_apply(lambda x: process.extractOne(x, list(crsp["COMNAM"]))[0])
# got warning message "Applied processor reduces input query to empty string, all comparisons will
# have score 0." which made me think i would get some NANs in mergecomnam but apparently it wasn't the
# case 
#%% drop helper column

se_us1 = se_us1.drop(columns = "SEHeadline_fix")
#%% count null values in the newly obtained column

se_us1["MergeComnam"].isnull().sum()
#%% save merged dataframe to csv

se_us1.to_csv("/home/edoardo/se_us2.csv")
#%% to illustrate some ideas about how to analyze the datasets for subsequent merging,
# i use a small and readable dataset containing earnings call information about the well known
# high tech company APPLE INC; the code below shows that the company was first inserted into CRSP in 1980
# as "APPLE COMPUTER INC", to then be inserted again in 2007 as "APPLE INC"; if the company had
# started to provide earnings calls in Q1 1981, up to 2020, the total amount of entries which
# should be visible in SE, directly related to the company, would then be (2020 - 1981) * 4 = 156
# this is not the case, since the first available data for apple's earnings call is Q2 2002, thus
# shrinking the total amount to (2020 - 2001) * 4 = 76 (since 2020 - 2002 would omit data for
# 1 year); this is almost the case, in fact when filtering for the company name, SE yields 75
# entries, in fact SE misses Q1 2002 information for the company; this implies that if,
# after merging, APPLE COMPUTER INC and APPLE INC appear, jointly, more than 75 times into SE
# MergeComnam, one or more mismatches occurred

se_apple = se[se["SECompanyName"].str.contains("Apple Inc")]

print(crsp["COMNAM"][crsp["PERMCO"] == 7])

print(crsp["DATE"][crsp["PERMCO"] == 7])
# more precisely, since data about apple appears for the first time in SE in 2002, and data for
# one call is missing, given that apple changed name at the beginning of 2007, 4 * 4 + 3 = 19
# entries should be matched with APPLE COMPUTER INC, while 75 - 19 = 56 should be matched with
# APPLE INC; this makes sense since from, and including, 2007 to 2020, 14 years passed, yielding
# 14 * 4 = 56 calls, which are all tracked into SE
