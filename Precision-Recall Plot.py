import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


total = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_All\Measures.csv",header=None,engine="python")

Nb = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_All\Classification Report.csv",header=None,engine="python")
Nb_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_50\Classification Report.csv",header=None,engine="python")
Nb_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_100\Classification Report.csv",header=None,engine="python")
Nb_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_1000\Classification Report.csv",header=None,engine="python")
Nb_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_10000\Classification Report.csv",header=None,engine="python")
Nb_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_20000\Classification Report.csv",header=None,engine="python")


Lr = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_All\Classification Report.csv",header=None,engine="python")
Lr_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_50\Classification Report.csv",header=None,engine="python")
Lr_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_100\Classification Report.csv",header=None,engine="python")
Lr_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_1000\Classification Report.csv",header=None,engine="python")
Lr_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_10000\Classification Report.csv",header=None,engine="python")
Lr_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_20000\Classification Report.csv",header=None,engine="python")


Rf = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_All\Classification Report.csv",header=None,engine="python")
Rf_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_50\Classification Report.csv",header=None,engine="python")
Rf_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_100\Classification Report.csv",header=None,engine="python")
Rf_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_1000\Classification Report.csv",header=None,engine="python")
Rf_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_10000\Classification Report.csv",header=None,engine="python")
Rf_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_20000\Classification Report.csv",header=None,engine="python")


print(Nb)

print((float(Nb.iloc[1][1])),Nb.iloc[1][2])


NBr = [float(Nb_50.iloc[2][2]),float(Nb_100.iloc[2][2]),float(Nb_1000.iloc[2][2]),float(Nb_10000.iloc[2][2]),float(Nb_20000.iloc[2][2]),float(Nb.iloc[2][2])]
NBp = [float(Nb_50.iloc[2][1]),float(Nb_100.iloc[2][1]),float(Nb_1000.iloc[2][1]),float(Nb_10000.iloc[2][1]),float(Nb_20000.iloc[2][1]),float(Nb.iloc[2][1])]

LRr = [float(Lr_50.iloc[2][2]),float(Lr_100.iloc[2][2]),float(Lr_1000.iloc[2][2]),float(Lr_10000.iloc[2][2]),float(Lr_20000.iloc[2][2]),float(Lr.iloc[2][2])]
LRp = [float(Lr_50.iloc[2][1]),float(Lr_100.iloc[2][1]),float(Lr_1000.iloc[2][1]),float(Lr_10000.iloc[2][1]),float(Lr_20000.iloc[2][1]),float(Lr.iloc[2][1])]

RFr = [float(Rf_50.iloc[2][2]),float(Rf_100.iloc[2][2]),float(Rf_1000.iloc[2][2]),float(Rf_10000.iloc[2][2]),float(Rf_20000.iloc[2][2]),float(Rf.iloc[2][2])]
RFp = [float(Rf_50.iloc[2][1]),float(Rf_100.iloc[2][1]),float(Rf_1000.iloc[2][1]),float(Rf_10000.iloc[2][1]),float(Rf_20000.iloc[2][1]),float(Rf.iloc[2][1])]



n = [50,100,1000,10000,20000,str(int(total.iloc[4][1]))]
i=0

plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot( NBr,NBp ,marker='o', color='#000099', label='Naive Bayes')
plt.plot( LRr, LRp,marker='o',color='#b30000', label='Logistic Regression')
plt.plot( RFr,RFp ,marker='o', color='#e68a00', label = 'Random Forest')


plt.grid()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('countVectorizer Tokenizer')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('countVectorizer Tokenizer Precision-Recall without legend',bbox_inches='tight')
plt.show()