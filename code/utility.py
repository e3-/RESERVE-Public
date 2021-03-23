import os
import shutil


class Dir_Structure:
    """Directory and file structure of the RESCUE model."""

    def __init__(self, code_dir=os.getcwd(), model_name='rescue'):
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
        self.data_dir = os.path.join(self.par_dir, "data", self.model_name)  # stores data/input
        self.output_dir = os.path.join(self.par_dir, "output", self.model_name)  # stores inferrence results
        self.logs_dir = os.path.join(self.par_dir, "logs", self.model_name)  # training log for tensorboard
        self.ckpts_dir = os.path.join(self.par_dir, "ckpts", self.model_name)  # check points for accidental pauses
        self.models_dir = os.path.join(self.par_dir, "models", self.model_name)  # trained models are stored here
        self.diag_dir = os.path.join(self.par_dir, "diagnostics", self.model_name)  # graphs and diagnostics

        # Define paths to files
        self.shuffled_indices_path = os.path.join(self.data_dir, "shuffled_indices_{}.npy".format(self.model_name))
        self.input_trainval_path = os.path.join(self.data_dir, "input_trainval.pkl")
        self.output_trainval_path = os.path.join(self.data_dir, "output_trainval.pkl")
        self.raw_data_path = os.path.join(self.raw_data_dir, "input_values_for_M_by_N_creating_script.csv")
        self.raw_data_validity_path = os.path.join(self.raw_data_dir, "input_validity_flags_for_M_by_N_creating_script.csv")
        self.pred_trainval_path = os.path.join(self.output_dir, "pred_trainval.pkl")
        self.training_hist_path = os.path.join(self.diag_dir, "training_history.npy")
        self.metrics_path = os.path.join(self.diag_dir, "metrics.npy")

        # clear all contents in the log directory
        if os.path.exists(self.logs_dir):
            shutil.rmtree(self.logs_dir)

        # make these directories if they do not already exist
        self.make_directories()

    def make_directories(self):
        for folder in [self.data_dir, self.raw_data_dir, self.output_dir,
                       self.logs_dir, self.ckpts_dir, self.models_dir, self.diag_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
