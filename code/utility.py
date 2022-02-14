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

from pathlib import Path
import shutil


class DirStructure:
    """Directory and file structure of the RESCUE model."""

    def __init__(self, code_dir=Path.cwd(), model_name="rescue"):
        """Initialize directory structure based on scenario name
        Args:
            code_dir (str): Path to `code` directory where all python code is located
            model_name (str): specific name of the model. Recommendation: RESCUE + VER number
        """
        self.model_name = model_name
        self.code_dir = code_dir

        # Define paths to directories
        self.par_dir = self.code_dir.parents[0]  # parent directory
        self.raw_data_dir = self.par_dir / "data" / "raw_data"
        # stores data checker outputs
        self.data_checker_dir = self.raw_data_dir / "data_checker_outputs"
        # stores extracted features
        self.data_dir = self.par_dir / "data" / self.model_name
        # stores extracted features
        self.output_dir = self.par_dir / "output" / self.model_name
        # training log for tensorboard
        self.logs_dir = self.par_dir / "logs" / self.model_name
        # check points for accidental pauses
        self.ckpts_dir = self.par_dir / "ckpts" / self.model_name
        # trained models are stored here
        self.models_dir = self.par_dir / "models" / self.model_name
        # diagnostics
        self.diag_dir = self.par_dir / "diagnostics" / self.model_name
        # diagnostic plots
        self.plots_dir = self.par_dir / "diagnostics" / self.model_name / "plots"

        # Define paths to files
        # TODO: separate the definition into the generic and specific portion
        self.RESERVE_input_path = self.raw_data_dir / "RESERVE_input_v1.xlsx"
        self.shuffled_indices_path = self.data_dir / "shuffled_indices_{}.npy".format(
            self.model_name
        )
        self.input_trainval_path = self.data_dir / "input_trainval.pkl"
        self.output_trainval_path = self.data_dir / "output_trainval.pkl"
        self.pred_trainval_path = self.output_dir / "pred_trainval.pkl"
        self.training_hist_path = self.diag_dir / "training_history.npy"
        self.metrics_path = self.diag_dir / "metrics.npy"

        # clear all contents in the log directory
        if Path.exists(self.logs_dir):
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
            Path.mkdir(folder, parents=True, exist_ok=True)
