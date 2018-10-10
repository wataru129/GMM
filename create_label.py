import os
import pandas as pd

#path = "/Users/wataru/Laboratry/reserch/test_extract/dataset/TUT-acoustic-scenes-2016-development/audio"
path = "/Users/wataru/Laboratry/reserch/GMM/dataset/data_car"

#files = os.listdir(path)
#files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
#print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']

files = []
files = os.listdir(path)

print(files)
df1 = pd.DataFrame(files)
df1 = df.reset_index(drop=True)
df1.shape
df2 = pd.DataFrame(df1.shape)
for i in range


#df = pd.concat([df1, df2, df3, df4])

df.to_csv('train.csv')
df
df2 = pd.DataFrame(files)
df2
