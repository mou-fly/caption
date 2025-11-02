import pandas as pd

from utils.h5 import load_h5

h5 = load_h5("C:\project\caption\data\processed_data_joint_30s_pods\data_train.h5")
data = h5['data'][1021]
device1 = data[1]

df = pd.DataFrame(device1, columns=['col1','col2','col3','col4','col5','col6'])

df.to_excel('output.xlsx', index=False)
print(device1.shape)



h5 = load_h5("C:\project\caption\data\WWADL_open\wifi\\0_1_1.h5")
print("hello")