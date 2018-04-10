import os
import pandas as pd

root_dir = os.getcwd() + '/Merged'

files = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]

files.sort()
files.remove('2016.xlsx')
df1 = pd.read_excel('/2016.xlsx')
for file in files:
    if file != ".DS_Store":
        df2 = pd.read_excel('Output/' + file)
        frames = [df1, df2]
        df1 = pd.concat(frames)

