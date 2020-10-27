import os
import pandas as pd
import numpy as np

## Constants
num_of_unique_file_types_to_be_created = 4  # Train inputs. train output, val inputs, val output
time_step_used_to_determine_date_of_sample = "T+1" # "T+1" is the time associated with response variable
day_block_indices_are_saved_with_name = "day_block_indices.npy"
train_inputs_filename_starts_with = "train_inputs_fold_"
train_output_filename_starts_with = "train_output_fold_"
val_inputs_filename_starts_with = "val_inputs_fold_"
val_output_filename_starts_with = "val_output_fold_"

## Master function
def get_train_and_val_sets_for_all_folds(path_to_trainval_inputs = None, path_to_trainval_output = None, path_to_trainval_datetimes = None,
                                         num_of_cv_folds = None, path_to_day_block_shuffled_indices = None,
                                         path_to_train_and_val_sets_for_all_folds = None, path_to_save_shuffled_indices = None,
                                         path_to_save_train_and_val_sets_for_all_folds = None):
    """
    Gets training and validation input and output sets  for each of the cross-validation folds. Either
    reads them in from a pre-determined, user-defined location or creates (and optionally saves) them
    :param path_to_trainval_inputs: Path to a MxN DataFrame holding trainval inputs. M is # of predictors and N is # of
                                    samples
    :param path_to_trainval_output: Path to a 1xN DataFrame holding trainval output
    :param path_to_trainval_datetimes: Path to a TxN DataFrame holding datetimes for each all time-points, "T" touched
                                       in a given training sample
    :param num_of_cv_folds: # of cross validation folds
    :param path_to_day_block_shuffled_indices: Path to a .npy holding previously defined day block indices if the user
                                               user wishes to recycle them
    :param path_to_train_and_val_sets_for_all_folds : Path to previously created train and val input and output datasets
                                                      (each a np.array), if the yser wishes to recycle them
    :param path_to_save_shuffled_indices: Path to save freshly created shuffled indices (as a .npy) if the user so wishes
    :param path_to_save_train_and_val_sets_for_all_folds: Path to save freshly created train and val inputs and outputs
                                                          (each as a .npy) for each fold if the user so wishes
    :return: X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds: Each is a list with num_of_cv_folds
             number of np.arrays holding one of train/val inputs/outputs for the fold # determined by the position
             of the array in the list
    """
    # If train and val sets have already been defined, simply read them in and then return them
    if path_to_train_and_val_sets_for_all_folds != None:
        print("Train and val sets are pre-determined. Proceeding to read them into memory....")
        X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds = read_predetermined_train_val_data(path_to_train_and_val_sets_for_all_folds)
        print("Train and val sets are ready!")
        return X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds

    # Else, create train and val sets
    else:
        # Read in trainval inputs and output data
        trainval_inputs_df = pd.read_pickle(path_to_trainval_inputs)
        trainval_output_df = pd.read_pickle(path_to_trainval_output)
        # If day shuffling is pre-determined, use that info
        if path_to_day_block_shuffled_indices != None:
            print("Day block shuffling pre-determnined....")
            day_block_shuffled_indices = np.load(os.path.join(path_to_day_block_shuffled_indices, day_block_indices_are_saved_with_name))
        # Else, perform day shuffling
        else:
            print("Performing day shuffling....")
            # Read in datetime data
            trainval_datetimes_df = pd.read_pickle(path_to_trainval_datetimes)
            # Get indices that will shuffle days
            day_block_shuffled_indices = create_and_shuffle_day_blocks(trainval_datetimes_df, path_to_save_shuffled_indices)
        # Apply to shuffle days in trainval dataset
        trainval_inputs_df = trainval_inputs_df.iloc[:, day_block_shuffled_indices]
        trainval_output_df = trainval_output_df.iloc[:, day_block_shuffled_indices]
        print("Done....")
        # Now split into train and val sets for each fold, to be returned to caller
        print("Creating train val sets for each fold....")
        X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds  = create_train_and_val_sets_for_each_fold(trainval_inputs_df,
                                                             trainval_output_df, num_of_cv_folds, path_to_save_train_and_val_sets_for_all_folds)
        print("Train and val sets are ready!")
    return X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds

## Helper functions
def read_predetermined_train_val_data(path_to_train_and_val_sets_for_all_folds):
    """
    Reads in and returns previously made train and val input and output np.arrays for each fold
    :param path_to_train_and_val_sets_for_all_folds: Path to the .npys to be read in
    :return: X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds: Each is a list with num_of_cv_folds
             number of np.arrays holding one of train/val inputs/outputs for the fold # determined by the position
             of the array in the list
    """
    # Taking stock of what's in the path
    num_of_cv_folds = int(len(os.listdir(path_to_train_and_val_sets_for_all_folds)) / num_of_unique_file_types_to_be_created)
    # To notify user of instances where he/she may have manually added/deleted some files
    if num_of_cv_folds != len(os.listdir(path_to_train_and_val_sets_for_all_folds)) / num_of_unique_file_types_to_be_created:
        print("Please confirm there are {} folds of data at the defined path. Proceeding with that assumption...".format(num_of_cv_folds))

    # Define collectors
    X_train_all_folds = []
    y_train_all_folds = []
    X_val_all_folds = []
    y_val_all_folds = []
    # Read in and fill collectors with data for each fold
    for cv_fold_idx in range(num_of_cv_folds):
        X_train_all_folds.append(np.load(os.path.join(path_to_train_and_val_sets_for_all_folds, train_inputs_filename_starts_with  + str(cv_fold_idx) + ".npy")))
        y_train_all_folds.append(np.load(os.path.join(path_to_train_and_val_sets_for_all_folds, train_output_filename_starts_with + str(cv_fold_idx) + ".npy")))
        X_val_all_folds.append(np.load(os.path.join(path_to_train_and_val_sets_for_all_folds, val_inputs_filename_starts_with + str(cv_fold_idx) + ".npy")))
        y_val_all_folds.append(np.load(os.path.join(path_to_train_and_val_sets_for_all_folds, val_output_filename_starts_with + str(cv_fold_idx) + ".npy")))
    print("Done reading train and val inputs and outputs for {} folds. Returning to caller....".format(num_of_cv_folds))

    return X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds

def create_and_shuffle_day_blocks(trainval_datetimes_df, path_to_save_shuffled_indices):
    """
    Identifies indices corresponding to each unique date present in the trainval dataset, and then shuffles indices
    such that days are shuffled among other days, while indices within a day are not.
    :param trainval_datetimes_df: pd.DataFrame containing datetimes corresponding to all trainval samples
    :param path_to_save_shuffled_indices: Path to save freshly created shuffled day block indices (as a .npy) if the
                                          user so wishes
    :return: day_block_shuffled_indices: np.array comprising of indices that shuffle days
    """
    # Extract dates from datetimes and identify unique dates
    all_trainval_datetimes = pd.to_datetime(trainval_datetimes_df.loc[time_step_used_to_determine_date_of_sample, :])
    all_trainval_dates = np.array(all_trainval_datetimes.dt.date.values)
    unique_trainval_dates = np.unique(all_trainval_dates)
    # Make a bucket for each unique date, fill it with sample indices belonging to that date
    list_of_unique_date_buckets = []
    for unique_dt in range(len(unique_trainval_dates)):
        list_of_unique_date_buckets.append(np.where(all_trainval_dates == unique_trainval_dates[unique_dt])[0])
    # Shuffle buckets, i.e days, and then stitch the shuffled days together into a single array
    np.random.shuffle(list_of_unique_date_buckets)
    day_block_shuffled_indices = np.concatenate(list_of_unique_date_buckets)

    # Save shuffled indices if user wishes
    if path_to_save_shuffled_indices != None:
        # Create a directory to store this file in, if it doesn't already exist
        if not os.path.isdir(path_to_save_shuffled_indices):
            os.makedirs(path_to_save_shuffled_indices)
        np.save(os.path.join(path_to_save_shuffled_indices, day_block_indices_are_saved_with_name), day_block_shuffled_indices)

    return day_block_shuffled_indices

def create_train_and_val_sets_for_each_fold(trainval_inputs_df, trainval_output_df, num_of_cv_folds,
                                            path_to_save_train_and_val_sets_for_all_folds):
    """
    Takes trainval data that has already been day shuffled. Partitions train val data into training and validation data
    based on current fold idx and total number of folds to be created. Optionally saves the train and val sets created.
    :param trainval_inputs_df: MxN DataFrame holding all training inputs, M being # of predictors and N being # of
                               trainval samples
    :param trainval_output_df: 1xN DataFrame holding all trainval output, N being the # of trainval samples
    :param num_of_cv_folds: Int: Total # of cross-validation folds
    :param path_to_save_train_and_val_sets_for_all_folds: Path to save train and val sets at. Optional.
    :return: X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds: Each is a list with num_of_cv_folds
             number of np.arrays holding one of train/val inputs/outputs for the fold # determined by the position
             of the array in the list
    """
    num_samples = trainval_inputs_df.shape[1]
    indices = np.arange(num_samples)

    # Define collectors
    X_train_all_folds = []
    y_train_all_folds = []
    X_val_all_folds = []
    y_val_all_folds = []

    for cv_fold_index in range(num_of_cv_folds):
        print("Creating training and validation sets for fold {} of {}".format(cv_fold_index + 1, num_of_cv_folds))
        # Define indices for training and validation set. Note that whole days have already been shuffled
        val_mask = np.zeros(len(indices), dtype=bool)
        val_mask[int(cv_fold_index / num_of_cv_folds * num_samples):int((cv_fold_index + 1) / num_of_cv_folds * num_samples)] = True
        val_indices = indices[val_mask]
        train_indices = indices[np.logical_not(val_mask)]

        # Shuffle indices
        np.random.shuffle(val_indices)
        np.random.shuffle(train_indices)

        # Extract and store train and val sets based on respective indices
        X_train_for_current_fold = trainval_inputs_df.iloc[:, train_indices].values.astype('float32').T
        y_train_for_current_fold = trainval_output_df.iloc[:, train_indices].values.astype('float32').T
        X_val_for_current_fold = trainval_inputs_df.iloc[:, val_indices].values.astype('float32').T
        y_val_for_current_fold = trainval_output_df.iloc[:, val_indices].values.astype('float32').T
        X_train_all_folds.append(X_train_for_current_fold)
        y_train_all_folds.append(y_train_for_current_fold)
        X_val_all_folds.append(X_val_for_current_fold)
        y_val_all_folds.append(y_val_for_current_fold)

        # Save train and val sets if user desires
        if path_to_save_train_and_val_sets_for_all_folds != None:
            # Create a directory to store these files in, if it doesn't already exist
            if not os.path.isdir(path_to_save_train_and_val_sets_for_all_folds):
                os.makedirs(path_to_save_train_and_val_sets_for_all_folds)
            np.save(os.path.join(path_to_save_train_and_val_sets_for_all_folds, train_inputs_filename_starts_with  + str(cv_fold_index) + ".npy"),
                                    X_train_for_current_fold)
            np.save(os.path.join(path_to_save_train_and_val_sets_for_all_folds, train_output_filename_starts_with + str(cv_fold_index) + ".npy"),
                                    y_train_for_current_fold)
            np.save(os.path.join(path_to_save_train_and_val_sets_for_all_folds, val_inputs_filename_starts_with + str(cv_fold_index) + ".npy"),
                                    X_val_for_current_fold)
            np.save(os.path.join(path_to_save_train_and_val_sets_for_all_folds, val_output_filename_starts_with + str(cv_fold_index) + ".npy"),
                                    y_val_for_current_fold)

    return X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds

# TODO: Lines below need to be transferred to rescue.ipynb

## User Inputs
# Names and paths to input and output files.
# Set these to None if you intend to simply read in training and validation sets for each fold that have already
# been created
trainval_inputs_file_name = "trainval_inputs.pkl"
trainval_output_file_name = "trainval_output.pkl"
trainval_datetimes_file_name = "trainval_datetimes.pkl"
path_to_trainval_inputs = os.path.join(os.getcwd(), "outputs_from_code", "trainval_M_by_N", trainval_inputs_file_name)
path_to_trainval_output = os.path.join(os.getcwd(), "outputs_from_code", "trainval_M_by_N", trainval_output_file_name)
path_to_trainval_datetimes = os.path.join(os.getcwd(), "outputs_from_code", "trainval_M_by_N", trainval_datetimes_file_name)
# Set this to None if you intend to create the day samples from scratch
path_to_day_block_shuffled_indices = None
# Set this to None if you intend to create fresh train and val sets
path_to_train_and_val_sets_for_all_folds = None
# Set these to None if you either aren't creating train and val sets or don't wish to save them for later
path_to_save_shuffled_indices = os.path.join(os.getcwd(), "outputs_from_code","day_block_shuffled_indices", "Created_20201026")
path_to_save_train_and_val_sets_for_all_folds = os.path.join(os.getcwd(), "outputs_from_code","train_and_val_sets_for_cv_folds", "Created_20201026")
# Set this to None if train and val sets have already been made and you are just reading them into memory
num_of_cv_folds = 10

# Call function
get_train_and_val_sets_for_all_folds(path_to_trainval_inputs, path_to_trainval_output, path_to_trainval_datetimes,
                                     num_of_cv_folds, path_to_day_block_shuffled_indices, path_to_train_and_val_sets_for_all_folds,
                                     path_to_save_shuffled_indices, path_to_save_train_and_val_sets_for_all_folds)

