import os
import pandas as pd
import numpy as np

## Constants
time_step_used_to_determine_date_of_sample = "T+1" # "T+1" is the time associated with response variable

## Master function
def get_CV_masks(trainval_datetimes, num_of_cv_folds = 10,
                 path_to_shuffled_indices = 'outputs_from_code/day_block_shuffled_indices.npy'):
    """
    Gets training and validation input and output sets  for each of the cross-validation folds. Either
    reads them in from a pre-determined, user-defined location or creates (and optionally saves) them
    :param trainval_datetimes: a TxN DataFrame holding datetimes for each all time-points, "T" touched
                                in a given training sample
    :param num_of_cv_folds: # of cross validation folds
    :param path_to_shuffled_indices: Path to save freshly created shuffled indices (as a .npy) if the user so wishes
    :return: train_indices_all_folds, val_indices_all_folds: Each is a list with num_of_cv_folds number of np.arrays 
            holding one of train/val indices for the fold # determined by the position of the array in the list
    """

    # If day shuffling is pre-determined, use that info
    if os.path.exists(os.path.join(path_to_shuffled_indices)):
        print("Day block shuffling pre-determnined....")
        day_block_shuffled_indices = np.load(path_to_shuffled_indices)
    # Else, perform day shuffling
    else:
        print("Performing day shuffling....")
        # Get indices that will shuffle days
        day_block_shuffled_indices = create_and_shuffle_day_blocks(trainval_datetimes, path_to_shuffled_indices)
    print("Done....")
    
    # Now split into train and val sets for each fold, to be returned to caller
    print("Creating train val masks for each fold....")
    val_masks_all_folds = create_val_masks_for_each_fold(day_block_shuffled_indices, num_of_cv_folds)
    print("Train and val masks are ready!")
    
    return val_masks_all_folds


def create_and_shuffle_day_blocks(trainval_datetimes, path_to_shuffled_indices):
    """
    Identifies indices corresponding to each unique date present in the trainval dataset, and then shuffles indices
    such that days are shuffled among other days, while indices within a day are not.
    :param trainval_datetimes: pd.DataFrame containing datetimes corresponding to all trainval samples
    :param path_to_shuffled_indices: Path to save freshly created shuffled day block indices (as a .npy) if the
                                          user so wishes
    :return: day_block_shuffled_indices: np.array comprising of indices that shuffle days
    """
    # Extract dates from datetimes and identify unique dates
    all_trainval_datetimes = pd.to_datetime(trainval_datetimes.loc[time_step_used_to_determine_date_of_sample, :])
    all_trainval_dates = np.array(all_trainval_datetimes.dt.date.values)
    unique_trainval_dates = np.unique(all_trainval_dates)
    # Make a bucket for each unique date, fill it with sample indices belonging to that date
    list_of_unique_date_buckets = []
    for unique_dt in range(len(unique_trainval_dates)):
        list_of_unique_date_buckets.append(np.where(all_trainval_dates == unique_trainval_dates[unique_dt])[0])
    # Shuffle buckets, i.e days, and then stitch the shuffled days together into a single array
    np.random.shuffle(list_of_unique_date_buckets)
    day_block_shuffled_indices = np.concatenate(list_of_unique_date_buckets)

    # Create a directory to store this file in, if it doesn't already exist
    if not os.path.isdir(os.path.dirname(os.path.abspath(path_to_shuffled_indices))):
        os.makedirs(os.path.dirname(os.path.abspath(path_to_shuffled_indices)))
    np.save(path_to_shuffled_indices, day_block_shuffled_indices)

    return day_block_shuffled_indices

def create_val_masks_for_each_fold(day_block_shuffled_indices, num_of_cv_folds):
    """
    Takes trainval data that has already been day shuffled. Partitions train val data into training and validation data
    based on current fold idx and total number of folds to be created. Optionally saves the train and val sets created.

    :param num_of_cv_folds: Int: Total # of cross-validation folds
    :param day_block_shuffled_indices: np.array comprising of indices that are shuffled as intra-day consecutive blocks
    :return: X_train_all_folds, y_train_all_folds, X_val_all_folds, y_val_all_folds: Each is a list with num_of_cv_folds
             number of np.arrays holding one of train/val inputs/outputs for the fold # determined by the position
             of the array in the list
    """
    num_samples = day_block_shuffled_indices.shape[0]
    indices = np.arange(num_samples)
    # Initialize containers to store masks determing the membership of the validation set 
    val_masks_all_folds = np.zeros((num_cv_folds,num_samples),dtype = bool)
    
    # iterate through every CV fold
    for cv_fold_index in range(num_of_cv_folds):
        print("Creating training and validation sets for fold {} of {}".format(cv_fold_index + 1, num_of_cv_folds))
        # Define indices for training and validation set. Note that whole days have already been shuffled
        val_mask = np.zeros(num_samples, dtype=bool)
        val_range_in_shuffled_indices = np.arange(int(cv_fold_index / num_of_cv_folds * num_samples),
                                                int((cv_fold_index + 1) / num_of_cv_folds * num_samples))
        val_mask[day_block_shuffled_indices[val_range_in_shuffled_indices]] = True
        val_masks_all_folds[cv_fold_index] = val_mask

    return val_masks_all_folds
