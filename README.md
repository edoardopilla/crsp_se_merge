# crsp_se_merge
Merging CRSP and SE datasets using Levenshtein distance based algorithm for fuzzy matching.

CRSP is a dataset containing information about american companies, often used for financial research, whereas SE is a dataset containing information related to earnings calls about companies around the world.

The problem which makes normal merging impossible is that company names change in time, for instance after mergers and acquisitions, so that usual merging of the two datasets would yield suboptimal results.

The column SEHeadline contains information related to company names which is consistent in time, and thus must be used jointly with the column COMNAM to merge the two datasets.

In the available .py file, non american companies are separated from american ones based on the assumption that the latter can be recognized by their legal entity, such as INC or CORP, although it must be clear that this approach isn't perfectly accurate, since some non american companies are also identified by the same legal entities, and at the same time some american companies may be identified by typically non american legal entities; this leads both to false positive and false negative matches in the merging process.

Further, some helper functions are defined to randomly pick dataframes of length 1000 on which the merging function is applied, to be then manually inspected in order to derive an estimate of the merging accuracy, which is found to be around 70% after checking 5 such dataframes. This approach can, in theory, lead to inspect overlapping subsets of the main dataframe, although the chance of slicing the same parts of the dataframe is low enough, at least within a single testing phase, not to warrant for an adjustment in the code, given that the overall dataframe length is, after separating for legal entities, beyond 300000 rows.

After having estimated the accuracy, the same function is then applied to the main dataframe for companies assumed to be american, and the results are saved to a csv file. For timing reasons, the code performs the step multiple times on a ~50000 row subset of the main dataframe, but in principle the function can be applied once to the whole dataframe. 
