import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


total = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_All\Measures.csv",header=None,engine="python")

P = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_All\Classification Report.csv",header=None,engine="python")
P_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_50\Classification Report.csv",header=None,engine="python")
P_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_100\Classification Report.csv",header=None,engine="python")
P_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_1000\Classification Report.csv",header=None,engine="python")
P_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_10000\Classification Report.csv",header=None,engine="python")
P_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_20000\Classification Report.csv",header=None,engine="python")


print(P)
print(P.iloc[1][1])
#---------------Precision---------
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[1][1]),float(P_100.iloc[1][1]),float(P_1000.iloc[1][1]),float(P_10000.iloc[1][1]),float(P_20000.iloc[1][1]),float(P.iloc[1][1])],linewidth=5, label='Negative')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[2][1]),float(P_100.iloc[2][1]),float(P_1000.iloc[2][1]),float(P_10000.iloc[2][1]),float(P_20000.iloc[2][1]),float(P.iloc[2][1])], color='red',label='Positive')
plt.xlabel("Number of Features")
plt.ylabel("Precision")
plt.title('CountVectorizer LemmaTokenizer Precision')
plt.legend()
plt.savefig('CountVectorizer LemmaTokenizer Precision',bbox_inches='tight')

#----------------Recall-----------
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[1][2]),float(P_100.iloc[1][2]),float(P_1000.iloc[1][2]),float(P_10000.iloc[1][2]),float(P_20000.iloc[1][2]),float(P.iloc[1][2])], label='Negative')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[2][2]),float(P_100.iloc[2][2]),float(P_1000.iloc[2][2]),float(P_10000.iloc[2][2]),float(P_20000.iloc[2][2]),float(P.iloc[2][2])], label='Positive')
plt.xlabel("Number of Features")
plt.ylabel("Recall")
plt.title('CountVectorizer LemmaTokenizer Recall')
plt.legend()
plt.savefig('CountVectorizer LemmaTokenizer Recall',bbox_inches='tight')


#---------------F1 score----------
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[1][0]),float(P_100.iloc[1][0]),float(P_1000.iloc[1][0]),float(P_10000.iloc[1][0]),float(P_20000.iloc[1][0]),float(P.iloc[1][0])],linewidth=8, label='Negative')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[2][0]),float(P_100.iloc[2][0]),float(P_1000.iloc[2][0]),float(P_10000.iloc[2][0]),float(P_20000.iloc[2][0]),float(P.iloc[2][0])],color='red', label='Positive')
plt.xlabel("Number of Features")
plt.ylabel("F1-score")
plt.title('CountVectorizer LemmaTokenizer F1-score')
plt.legend()
plt.savefig('CountVectorizer LemmaTokenizer F1-score',bbox_inches='tight')

#---------------Weighted average----------
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,total.iloc[4][1]],[float(P_50.iloc[3][0]),float(P_100.iloc[3][0]),float(P_1000.iloc[3][0]),float(P_10000.iloc[3][0]),float(P_20000.iloc[3][0]),float(P.iloc[1][0])])
plt.xlabel("Number of Features")
plt.ylabel("Weighted Average")
plt.title('CountVectorizer LemmaTokenizer Weighted Average')
plt.savefig('CountVectorizer LemmaTokenizer Weighted Average',bbox_inches='tight')