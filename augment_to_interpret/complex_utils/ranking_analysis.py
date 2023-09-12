"""
Contain code to analyse rankings.
It was used to analyse node feature selection, but it is not used anymore.
"""

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


def ranking_regression_analysis(
    source_ranking,  # ie. unsupervised
    reference_ranking,  # ie. supervised
    background_ranking  # ie. entropy
):
    """
    Input : three vectors of rankings, such as [1,12,5,6,...] and [3,5,6,...]

    Asserts whether the source ranking is significantly different from the bakcground ranking.
    Also checks whether there is a sufficient relation between the source, reference and background ranking so that the reference ranking can be
    recovered from the source.

    To be used like this : our unsupervised ranking is the source, gsat is the reference ranking, entropy ranking is the background ranking
    """
    result_dict = {}

    # Kendall tau (1 is strong agreement, -1 is strong disagreement)
    tau, p_value = stats.kendalltau(source_ranking, reference_ranking)
    result_dict["agreement_source_reference"] = tau
    result_dict["agreement_source_reference_pval"] = p_value

    # plt.figure() ; plt.scatter(source_ranking, reference_ranking)

    tau, p_value = stats.kendalltau(source_ranking, background_ranking)
    result_dict["agreement_source_background"] = tau
    result_dict["agreement_source_background_pval"] = p_value

    # plt.figure() ; plt.scatter(source_ranking, background_ranking)

    # Can refernce be predicted with source and background ?
    x = np.concatenate((
        source_ranking.reshape(-1, 1),
        background_ranking.reshape(-1, 1)
    ),
        axis=1)
    y = reference_ranking
    reg = LinearRegression().fit(x, y)
    result_dict["reference_predictability"] = reg.score(x, y)
    result_dict["reference_predictability_coef"] = reg.coef_
    result_dict["predicted_reference_from_source_and_background"] = reg.predict(x)

    # plt.figure(); plt.scatter(
    #     result_dict["predicted_reference_from_source_and_background"],
    #     reference_ranking
    # )

    # TODO Try to predict supervised from entropy, and then from entropy+unsup to show unsup brings improvement

    return result_dict
