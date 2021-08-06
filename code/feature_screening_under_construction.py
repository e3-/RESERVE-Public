# Script is meant to screen input features to ensure we can avoid redundancy/noise when training the model
# Not actively used at present. Needs to be completed and linked to the rest of data processing before more features
# are introduced

# ############################ LICENSE INFORMATION ############################
# This file is part of the E3 RESERVE Model.

# Copyright (C) 2021 Energy and Environmental Economics, Inc.
# For contact information, go to www.ethree.com

# The E3 RESERVE Model is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The E3 RESERVE Model is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with the E3 RESERVE Model (in the file LICENSE.TXT). If not,
# see <http://www.gnu.org/licenses/>.
# #############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs to be used in correlation analysis
# Directions associated whether we wish to retain any features by default irrespective of correlation results
mandatory_feature = [True, True, True, True, True, True]
# How many additional features to add in based on correlation analysis?
max_num_optional_features = 0


def identify_highly_correlated_features(
    df, optional_feature, max_num_optional_features
):
    """
    Calculates and plots pair-wise correlation matrix for each column in df. Then identifies most highly
    correlated optional features (each represented by a column in the df) to remove
    :param df: DataFrame - each column holds values corresponding to one feature
    :param optional_feature: Array holding booleans, length and order corresponds to column names in df. True if
    a particular column is optional and can be dropped if found to be highly correlated with the others
    :param max_num_optional_features: Int-How many of the optional features can be retained?
    :return: features_to_remove: List of features to remove from ML inputs' data
    """
    # Calculate and plot correlation matrix
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap="Reds")
    plt.colorbar()
    plt.yticks(np.arange(len(corr_matrix.index)), corr_matrix.index)
    plt.xticks(
        np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation="vertical"
    )
    plt.title("Correlation Matrix For All Features", fontweight="bold")
    plt.show()
    # Of the optional features, identify the ones most correlated with the others
    optional_features_corr_sum = (
        corr_matrix.sum(axis=1).loc[optional_feature].sort_values()
    )
    features_to_remove = optional_features_corr_sum.index[max_num_optional_features:]
    return features_to_remove


# This will be used to keep account of which features were turned into multiple ones due to misalignment in time-step
final_feature_count = np.ones_like(mandatory_feature, dtype=int)
feature_idx = 0
##### Iterate over each feature output from data-checker #####
# Make note of the fact that we've created additional features
num_substeps = int(ML_inputs_time_step / feature_time_step)
final_feature_count[feature_idx] = num_substeps
feature_idx += 1

# Ensure newly created features get tagged as optional/mandatory as appropriate
mandatory_feature = np.repeat(mandatory_feature, final_feature_count)
optional_feature = np.logical_not(mandatory_feature)
# Conduct correlation analysis and identify features to be removed, if any
print(
    "Performing correlation analysis to remove optional features highly correlated with the others...."
)
features_to_remove = identify_highly_correlated_features(
    ML_inputs_df, optional_feature, max_num_optional_features
)
print("Features to be removed- {}....".format(features_to_remove.values))
# Drop these feature from ML inputs' df
ML_inputs_df = ML_inputs_df.drop(columns=features_to_remove)
ML_inputs_validity_df = ML_inputs_validity_df.drop(columns=features_to_remove)
