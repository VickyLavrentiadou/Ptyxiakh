import pandas as pd
import numpy as np
import gzip
import csv



def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


df = getDF(r'C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\data\reviews_Apps_for_Android_5.json.gz')
df = df[df['overall']!= 3]
df['Ratings'] = np.where(df['overall']>3, 1, 0)

to_remove1 = np.random.choice(df[df['Ratings']==0].index,size=103098,replace=False)
df=df.drop(to_remove1)

to_remove2 = np.random.choice(df[df['Ratings']==1].index,size=524718,replace=False)
df=df.drop(to_remove2)
print(df)
df.to_csv('MyData.csv')