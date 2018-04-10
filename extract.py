import os
import pandas as pd

root_dir = os.getcwd() + '/Data/BasicStats/'
dir2 = os.getcwd() + '/Data/Advanced/'

files = [item for item in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, item))]
files_adv = [item for item in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, item))]

files.sort()
files_adv.sort()

for file in files:
    if file != ".DS_Store":
        df = pd.read_csv('Data/Advanced/' + file, encoding='utf-8')
        df2 = pd.read_csv('Data/BasicStats/' + file, encoding='utf-8')
        df = df[df["School"].str.contains("NCAA")]
        df2 = df2[df2["School"].str.contains("NCAA")]
        result = pd.concat([df, df2], axis=1, join='inner')
        result = result.loc[:,~result.columns.duplicated()]
        split = file.split('.')
        writer = pd.ExcelWriter('Output/Advanced/' + split[0] + ".xlsx")
        result.to_excel(writer,'Sheet1')
        writer.save()
