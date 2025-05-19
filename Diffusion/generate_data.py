import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from IELDiffusion import XELDiffusionModel
import numpy as np
import warnings

warnings.filterwarnings('ignore')
label_encoder = LabelEncoder()

dataset = 'MZVAV21'
method = 'vp'
model = 'xgb'
num = 50
fault_data = []
num_g = 7000
data = pd.read_csv(
    "D:/code/IELDM/datasets/" + str(dataset) + "/" + str(dataset) + "_select_" + str(num) + ".csv")

if dataset == 'chiller':
    fault_data = data[data['Y'] != 8]
    num_g = 7000
if dataset == 'MZVAV21':
    fault_data = data[data['Y'] != 4]
    num_g = 3000
if dataset == 'SZCAV':
    fault_data = data[data['Y'] != 15]
    num_g = 7000

X = fault_data.iloc[:, fault_data.columns != 'Y'].to_numpy()
y = fault_data.iloc[:, fault_data.columns == 'Y']
y = pd.DataFrame(label_encoder.fit_transform(y), columns=['Y']).to_numpy().ravel()

print('Training...')
start = time.time()
forest_model = XELDiffusionModel(X=X, label_y=y, n_t=50, model=model, duplicate_K=100,
                                    bin_indexes=[], cat_indexes=[], int_indexes=[], p_in_one=True,
                                    diffusion_type='vp', n_batch=0, gpu_hist=False, n_jobs=-1)
train_end = time.time()
print('Training.time: %.2f' % (train_end-start))
print('Generating...')
Xy_fake = forest_model.generate(num_g)
end = time.time()
file_path = f"generate_data/{dataset}/{method}/{model}/fake_{num}_cpu_{model}.csv"
np.savetxt(file_path, Xy_fake, delimiter=',')
print('Generation.time: %.2f' % (end - train_end))

time_file = f"generate_data/{dataset}/{method}/time/time_{num}_cpu_{model}.txt"
with open(time_file, 'w') as file:
    file.write(f"Training time:\n{train_end-start}\n")
    file.write(f"Generation time:\n{end - train_end}\n")

