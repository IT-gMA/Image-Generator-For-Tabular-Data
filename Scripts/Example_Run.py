import pandas as pd
import os.path
import os
import argparse
from IGTD_Functions import min_max_transform, table_to_image


# Retrieve the tabular data file's name in the input argument line
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Data.txt',
                        help='Tabular data files name (saved in ../Data/) to be converted to image')
    parser.add_argument('--result', type=str, default='Results',
                        help='Directory to save the recent runs')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


opt = parse_opt(True)
# Verify whether the input tabular data file exists
tab_data_path = '../Data/Data.txt'
if opt.data:
    tab_data_path = '../Data/{}'.format(opt.data)
    if not os.path.exists(tab_data_path):
        raise Exception("The tabular data can't be found")

# Verify the integrity of the Result storing folder
saved_result_dir = '../Results/'
if opt.result:
    saved_result_dir = '../{}/'.format(opt.result)
    # Create this new directory if it does not exist yet
    if not os.path.exists(tab_data_path):
        os.makedirs(saved_result_dir, 0)
    else:
        # Check whether this directory is empty in case it already exists
        if os.listdir(saved_result_dir):
            raise Exception("This directory already has datas in it")

num_row = 3  # Number of pixel rows in image representation
num_col = 4  # Number of pixel columns in image representation
# num_row x num_col ~= num_features?, feature pixels are tightly packed

num = num_row * num_col  # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 2  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000  # The maximum number of iterations to run the IGTD algorithm, if it does not converge. S_max
val_step = 600  # The number of iterations for determining algorithm convergence. If the error reduction rate
# is smaller than a pre-set threshold for val_step itertions, the algorithm converges.  S_con

# BATCH NORMALISATION: Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data = pd.read_csv(tab_data_path, low_memory=False, sep='\t', engine='c', na_values=['na', '-', ''],
                   header=0, index_col=0)
data = data.iloc[:, :num]  # Averaging over num (number of features in each sample), use this for normalisation below
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'  # (1) feature distance
image_dist_method = 'Euclidean'  # (2) pixel distance
error = 'abs'  # difference between the feature distance and pixel distance ranking matrices
# (can either take the absolute or square value)
result_dir = '{}Test_1'.format(saved_result_dir)  # Where we save the computed difference (results)
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)  # Using the IGTD's function

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'  # Distance (or difference kinda) between features (smaller)
image_dist_method = 'Manhattan'  # Difference between the matrices (containing features or pixels) (bigger)
error = 'squared'
result_dir = '{}Test_2'.format(saved_result_dir)
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
