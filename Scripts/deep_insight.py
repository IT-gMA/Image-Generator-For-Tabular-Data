import pandas as pd
import os.path
import os
import argparse
import numpy as np
from pyDeepInsight import ImageTransformer, Norm2Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import cv2
import matplotlib.pyplot as plt

NUM_ROW = 40
NUM_COLUMN = 40
SAVE_IMG_PATH = '../DI_bleve_40x40'
SAVE_LOG_PATH = '../DI_bleve_40x40/run_logs'


from IGTD_Functions import min_max_transform, table_to_image

tab_data_path = '../Data/new_data_40.txt'
data = pd.read_csv(tab_data_path, low_memory=False, sep='\t',
                   header=0, index_col=0)
num = NUM_ROW*NUM_COLUMN
data = data.iloc[:, 0:]  # Averaging over num (number of features in each sample), use this for normalisation below
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

if isinstance(norm_data, pd.DataFrame):
    samples = norm_data.index.map(str)
    norm_data = norm_data.values
else:
    samples = [str(i) for i in range(norm_data.shape[0])]

print(norm_data)

samples_temp = []
for sample in samples:
    samples_temp.append(sample)

samples = samples_temp


ln = Norm2Scaler()
data_train_norm = ln.fit_transform(norm_data)

le = LabelEncoder()
label_encoded = le.fit_transform(samples)

distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    init='random',
    n_jobs=-1
)

pixel_size = (NUM_ROW*3, NUM_COLUMN*3)
it = ImageTransformer(
    feature_extractor=reducer,
    pixels=pixel_size)

it.fit(norm_data, y=None, plot=True)
generated_imgs = it.transform(data_train_norm)
fdm = it.feature_density_matrix()
fdm[fdm == 0] = np.nan

plt.figure(figsize=(10, 7.5))
plt.savefig(fname=SAVE_LOG_PATH + "/feature_density_matrix.png", bbox_inches='tight', pad_inches=0)


for i in range(len(generated_imgs)):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(generated_imgs[i], cmap='gray', vmin=0, vmax=255)
    plt.savefig(fname=SAVE_IMG_PATH + '/_' + samples[i] + '_image.png', bbox_inches='tight')
    plt.close(fig)

ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.,
                 linecolor="lightgrey", square=True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

for _, spine in ax.spines.items():
    spine.set_visible(True)
_ = plt.title("BLEVE data point per pixel")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.savefig(fname=SAVE_LOG_PATH + "/BLEVE data point per pixel.png", bbox_inches='tight', pad_inches=0)
plt.show()
