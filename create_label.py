import os
import pandas as pd

path = "/Users/wataru/Laboratry/reserch/test_extract/dataset/TUT-acoustic-scenes-2016-development/audio"

files = os.listdir(path)
files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
#print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']


index = ["a"]
#df = pd.read_csv('some.csv')
df = pd.Series(files_file[0], index= index)
print(df)



df1 = pd.read_table("/Users/wataru/Laboratry/reserch/test_extract/dataset/TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_train.txt", delim_whitespace=True, header=None)
df2 = pd.read_table("/Users/wataru/Laboratry/reserch/test_extract/dataset/TUT-acoustic-scenes-2016-development/evaluation_setup/fold2_train.txt", delim_whitespace=True, header=None)
df3 = pd.read_table("/Users/wataru/Laboratry/reserch/test_extract/dataset/TUT-acoustic-scenes-2016-development/evaluation_setup/fold3_train.txt", delim_whitespace=True, header=None)
df4 = pd.read_table("/Users/wataru/Laboratry/reserch/test_extract/dataset/TUT-acoustic-scenes-2016-development/evaluation_setup/fold4_train.txt", delim_whitespace=True, header=None)
#print(df1)
#print(df2)
#print(df3)
#print(df4)

df = pd.concat([df1, df2, df3, df4])
df = df.reset_index(drop=True)
df.to_csv('train.csv')
df

a = ["a","b"]
a
