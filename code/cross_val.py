import os
import pandas as pd
import numpy as np


## Master function
def get_CV_masks(trainval_datetimes, num_of_cv_folds, path_to_shuffled_indices):

    """
    Gets the masks determining what data goes into validation for each of the cross-validation folds. The shuffled
    indices are either read from a pre-determined file or created ad-hoc.
    :param trainval_datetimes: a length N DataFrame holding datetimes for each sample. Totalling N samples
    :param num_of_cv_folds: # of cross validation folds
    :param path_to_shuffled_indices: Path to save/read shuffled indices (as a .npy)
    :return: val_masks_all_folds: A boolean 2D np array. The first dimension corresponds to the num_of_cv_folds,
             while the 2nd dimension is for each sample. True means this sample belongs to the validation set
             in this cross validation fold.
    """

    # If day shuffling is pre-determined, use that info
    if os.path.exists(path_to_shuffled_indices):
        print("Day block shuffling pre-determnined....")
        day_block_shuffled_indices = np.load(path_to_shuffled_indices)
    # Else, perform day shuffling
    else:
        print("Performing day shuffling....")
        # Get indices that will shuffle days
        day_block_shuffled_indices = create_and_shuffle_day_blocks(
            trainval_datetimes, path_to_shuffled_indices
        )
    print("Done....")

    # Now split into train and val sets for each fold, to be returned to caller
    print("Creating train val masks for each fold....")
    val_masks_all_folds = create_val_masks_for_each_fold(
        day_block_shuffled_indices, num_of_cv_folds
    )
    print("Train and val masks are ready!")

    return val_masks_all_folds


def create_and_shuffle_day_blocks(trainval_datetimes, path_to_shuffled_indices):
    """
    Identifies indices corresponding to each unique date present in the trainval dataset, and then shuffles indices
    such that days are shuffled among other days, while indices within a day are not.
    :param trainval_datetimes: pd.DataFrame containing datetimes corresponding to all trainval samples
    :param path_to_shuffled_indices: Path to save shuffled day block indices (as a .npy)
    :return: day_block_shuffled_indices: np.array comprising of indices that shuffle days
    """
    # Extract dates from datetimes and identify unique dates
    all_trainval_dates = trainval_datetimes.date
    unique_trainval_dates = np.unique(all_trainval_dates)
    # Make a bucket for each unique date, fill it with sample indices belonging to that date
    list_of_unique_date_buckets = []
    for unique_dt in range(len(unique_trainval_dates)):
        list_of_unique_date_buckets.append(
            np.where(all_trainval_dates == unique_trainval_dates[unique_dt])[0]
        )
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
    based on current fold idx and total number of folds to be created.

    :param num_of_cv_folds: Int: Total # of cross-validation folds
    :param day_block_shuffled_indices: np.array comprising of indices that are shuffled as intra-day consecutive blocks
    :return: val_masks_all_folds: A boolean 2D np array. The first dimension corresponds to the num_of_cv_folds,
             while the 2nd dimension is for each sample. True means this sample belongs to the validation set
             in this cross validation fold.
    """
    num_samples = day_block_shuffled_indices.shape[0]
    indices = np.arange(num_samples)
    # Initialize containers to store masks determing the membership of the validation set
    val_masks_all_folds = np.zeros((num_of_cv_folds, num_samples), dtype=bool)

    # iterate through every CV fold
    for cv_fold_index in range(num_of_cv_folds):
        # Define indices for training and validation set. Note that whole days have already been shuffled
        val_mask = np.zeros(num_samples, dtype=bool)
        val_range_in_shuffled_indices = np.arange(
            int(cv_fold_index / num_of_cv_folds * num_samples),
            int((cv_fold_index + 1) / num_of_cv_folds * num_samples),
        )
        val_mask[day_block_shuffled_indices[val_range_in_shuffled_indices]] = True
        val_masks_all_folds[cv_fold_index] = val_mask

    return val_masks_all_folds
