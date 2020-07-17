import pandas as pd
import os
import numpy as np
folder_name=os.listdir('data/original_data')
folder_name_filtered=[s for s in folder_name if 'csv' in s]
full_data=pd.DataFrame()
for csv_file in folder_name_filtered:
    tmp_df=pd.read_csv("data/original_data/"+csv_file,header=0,index_col=False)
    print("load"+" data/original_data/"+csv_file)
    full_data=full_data.append(tmp_df)
full_data.reset_index(drop=True,inplace=True)
map={1:0,2:1,3:1,4:2,5:2}
full_data['score']=full_data['score'].map(map)

train_index=np.random.choice(full_data.index,size=int(0.8*len(full_data)),replace=False)
train_data=full_data.iloc[train_index].copy()
test_data=full_data.drop(full_data.index[train_index])

train_data.to_csv('data/train_data.csv',encoding='utf-8-sig',index=False)
test_data.to_csv('data/test_data.csv',encoding='utf-8-sig',index=False)
