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
import shutil


class Dir_Structure:
    """Directory and file structure of the RESCUE model."""

    def __init__(self, code_dir=os.getcwd(), model_name="rescue"):
        """Initialize directory structure based on scenario name
        Args:
            code_directory (str): Path to `code` directory where all python code is located
            model_name (str): specific name of the model. Recommendation: RESCUE + VER number
        """
        self.model_name = model_name
        self.code_dir = code_dir

        # Define paths to directories
        self.par_dir = os.path.dirname(self.code_dir)  # parent directory
        self.raw_data_dir = os.path.join(self.par_dir, "data", "raw_data")
        self.data_checker_dir = os.path.join(
            self.raw_data_dir, "data_checker_outputs"
        )  # stores data checker outputs
        self.data_dir = os.path.join(
            self.par_dir, "data", self.model_name
        )  # stores data/input
        self.output_dir = os.path.join(
            self.par_dir, "output", self.model_name
        )  # stores inference results
        self.logs_dir = os.path.join(
            self.par_dir, "logs", self.model_name
        )  # training log for tensorboard
        self.ckpts_dir = os.path.join(
            self.par_dir, "ckpts", self.model_name
        )  # check points for accidental pauses
        self.models_dir = os.path.join(
            self.par_dir, "models", self.model_name
        )  # trained models are stored here
        self.diag_dir = os.path.join(
            self.par_dir, "diagnostics", self.model_name
        )  # diagnostics
        self.plots_dir = os.path.join(
            self.par_dir, "diagnostics", self.model_name, "plots"
        )  # diagnostic plots

        # Define paths to files
        self.RESERVE_settings_path = os.path.join(self.par_dir, "data", "RESERVE_settings.xlsx")
        self.shuffled_indices_path = os.path.join(
            self.data_dir, "shuffled_indices_{}.npy".format(self.model_name)
        )
        self.input_trainval_path = os.path.join(self.data_dir, "input_trainval.pkl")
        self.output_trainval_path = os.path.join(self.data_dir, "output_trainval.pkl")
        self.pred_trainval_path = os.path.join(self.output_dir, "pred_trainval.pkl")
        self.training_hist_path = os.path.join(self.diag_dir, "training_history.npy")
        self.metrics_path = os.path.join(self.diag_dir, "metrics.npy")

        # clear all contents in the log directory
        if os.path.exists(self.logs_dir):
            shutil.rmtree(self.logs_dir)

        # make these directories if they do not already exist
        self.make_directories()

    def make_directories(self):
        for folder in [
            self.data_dir,
            self.raw_data_dir,
            self.output_dir,
            self.logs_dir,
            self.ckpts_dir,
            self.models_dir,
            self.diag_dir,
            self.plots_dir,
        ]:
            if not os.path.exists(folder):
                os.makedirs(folder)
