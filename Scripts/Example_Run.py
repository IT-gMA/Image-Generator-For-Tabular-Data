import pandas as pd
import os
from IGTD_Functions import min_max_transform, table_to_image



num_row = 30    # Number of pixel rows in image representation
num_col = 30    # Number of pixel columns in image representation
# num_row x num_col ~= num_features?, feature pixels are tightly packed

num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge. S_max
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.  S_con

# BATCH NORMALISATION: Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data = pd.read_csv('../Data/BLEVE_data.txt', low_memory=False, sep='\t', engine='c', na_values=['na', '-', ''],
                header=0, index_col=0)
data = data.iloc[:, :num]       # Averaging over num (number of features in each sample), use this for normalisation below
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'           # (1) feature distance
image_dist_method = 'Euclidean'         # (2) pixel distance
error = 'abs'                           # difference between the feature distance and pixel distance ranking matrices
                                        # (can either take the absolute or square value)
result_dir = '../Results/Test_1'        # Where we save the computed difference (results)
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)       # Using the IGTD's function

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'         # Distance (or difference kinda) between features (smaller)
image_dist_method = 'Manhattan'     # Difference between the matrices (containing features or pixels) (bigger)
error = 'squared'
result_dir = '../Results/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
