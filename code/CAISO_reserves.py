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

# The model for CAISO is unique. We predict reserves on a 5-min basis. We then find the largest reserve level in each
# 15-min interval to be applicable throughout that 15-min interval. Code in this cell executes this idea and yields
# 15-min reserve levels. It will also modify output_trainval and val(idation)_masks accordingly to ensure that
# metrics calculation and diagnostic plotting can take place downstream

import numpy as np
import pandas as pd

RTPD_interval = pd.Timedelta("15T")
RTD_interval = pd.Timedelta("5T")


def get_CAISO_RTPD_reserve(pred_trainval):
    # Find which rows belong to the same RTPD interval
    anchor_dt = pd.Timestamp(pred_trainval.index[0].year, 1, 1)
    RTPD_interval_idx = (pred_trainval.index - anchor_dt) // RTPD_interval

    # According to CAISO methodology, headroom take the maximum of the sub intervals,
    # footroom take the min, which Median forecast bias take the average

    pred_trainval_RTPD_max = pred_trainval.groupby(RTPD_interval_idx).max()
    pred_trainval_RTPD_min = pred_trainval.groupby(RTPD_interval_idx).min()
    pred_trainval_RTPD_mean = pred_trainval.groupby(RTPD_interval_idx).mean()

    PI_percentiles = pred_trainval.columns.levels[0]
    footroom_percentiles = [PI for PI in PI_percentiles if PI < 0.5]

    pred_trainval_RTPD = pred_trainval_RTPD_max.copy()
    pred_trainval_RTPD[pd.IndexSlice[footroom_percentiles]] = pred_trainval_RTPD_min[
        pd.IndexSlice[footroom_percentiles]
    ].values
    pred_trainval_RTPD[pd.IndexSlice[0.5]] = pred_trainval_RTPD_mean[
        pd.IndexSlice[0.5]
    ].values

    # restore original DT index
    pred_trainval_RTPD.index = pred_trainval_RTPD.index * RTPD_interval + anchor_dt

    # resample into the orignal RTD interval, but only with the RTPD values
    pred_trainval_RTPD = pred_trainval_RTPD.resample(RTD_interval).pad()

    # Only keep the datetimes that have appeared in the original
    pred_trainval_RTPD = pred_trainval_RTPD.loc[pred_trainval.index]

    return pred_trainval_RTPD
