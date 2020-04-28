import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#----------------countVectorizer Tokenizer---------------------------
NB = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_All\Measures.csv",header=None,engine="python")
NB_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_50\Measures.csv",header=None,engine="python")
NB_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_100\Measures.csv",header=None,engine="python")
NB_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_1000\Measures.csv",header=None,engine="python")
NB_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_10000\Measures.csv",header=None,engine="python")
NB_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_Tokenizer_20000\Measures.csv",header=None,engine="python")


LR = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_All\Measures.csv",header=None,engine="python")
LR_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_50\Measures.csv",header=None,engine="python")
LR_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_100\Measures.csv",header=None,engine="python")
LR_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_1000\Measures.csv",header=None,engine="python")
LR_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_10000\Measures.csv",header=None,engine="python")
LR_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_Tokenizer_20000\Measures.csv",header=None,engine="python")


RF = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_All\Measures.csv",header=None,engine="python")
RF_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_50\Measures.csv",header=None,engine="python")
RF_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_100\Measures.csv",header=None,engine="python")
RF_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_1000\Measures.csv",header=None,engine="python")
RF_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_10000\Measures.csv",header=None,engine="python")
RF_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_Tokenizer_20000\Measures.csv",header=None,engine="python")


print(NB)

#------------------------Classifier Accuracy Plot-----
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,NB.iloc[4][1]],[NB_50.iloc[2][1],NB_100.iloc[2][1],NB_1000.iloc[2][1],NB_10000.iloc[2][1],NB_20000.iloc[2][1],NB.iloc[2][1]],linewidth=8, label='Naive Bayes')
plt.plot([50,100,1000,10000,20000,LR.iloc[4][1]],[LR_50.iloc[2][1],LR_100.iloc[2][1],LR_1000.iloc[2][1],LR_10000.iloc[2][1],LR_20000.iloc[2][1],LR.iloc[2][1]], label='Logistic Reression')
plt.plot([50,100,1000,10000,20000,RF.iloc[4][1]],[RF_50.iloc[2][1],RF_100.iloc[2][1],RF_1000.iloc[2][1],RF_10000.iloc[2][1],RF_20000.iloc[2][1],RF.iloc[2][1]], color='red',label='Random Forest')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title('CountVectorizer Tokenizer')
plt.legend()
plt.savefig('CountVectorizer Tokenizer Accuracy',bbox_inches='tight')


#------------------------------------------------------------------------------------------------------
#---------------------CountVectorizer LemmaTokenizer------------------------
NB = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_All\Measures.csv",header=None,engine="python")
NB_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_50\Measures.csv",header=None,engine="python")
NB_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_100\Measures.csv",header=None,engine="python")
NB_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_1000\Measures.csv",header=None,engine="python")
NB_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_10000\Measures.csv",header=None,engine="python")
NB_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_20000\Measures.csv",header=None,engine="python")


LR = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_All\Measures.csv",header=None,engine="python")
LR_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_50\Measures.csv",header=None,engine="python")
LR_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_100\Measures.csv",header=None,engine="python")
LR_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_1000\Measures.csv",header=None,engine="python")
LR_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_10000\Measures.csv",header=None,engine="python")
LR_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_20000\Measures.csv",header=None,engine="python")


RF = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_All\Measures.csv",header=None,engine="python")
RF_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_50\Measures.csv",header=None,engine="python")
RF_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_100\Measures.csv",header=None,engine="python")
RF_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_1000\Measures.csv",header=None,engine="python")
RF_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_10000\Measures.csv",header=None,engine="python")
RF_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_20000\Measures.csv",header=None,engine="python")


#------------------------Classifier Accuracy Plot-----
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,NB.iloc[4][1]],[NB_50.iloc[2][1],NB_100.iloc[2][1],NB_1000.iloc[2][1],NB_10000.iloc[2][1],NB_20000.iloc[2][1],NB.iloc[2][1]],linewidth=8, label='Naive Bayes')
plt.plot([50,100,1000,10000,20000,LR.iloc[4][1]],[LR_50.iloc[2][1],LR_100.iloc[2][1],LR_1000.iloc[2][1],LR_10000.iloc[2][1],LR_20000.iloc[2][1],LR.iloc[2][1]], label='Logistic Reression')
plt.plot([50,100,1000,10000,20000,RF.iloc[4][1]],[RF_50.iloc[2][1],RF_100.iloc[2][1],RF_1000.iloc[2][1],RF_10000.iloc[2][1],RF_20000.iloc[2][1],RF.iloc[2][1]],color = 'red', label='Random Forest')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title('CountVectorizer LemmaTokenizer')
plt.legend()
plt.savefig('CountVectorizer LemmaTokenizer Accuracy',bbox_inches='tight')


#----------------------------------------------------------------------------------------------------
#---------------------Tfidf Tokenizer------------------------
NB = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_Tokenizer_All\Measures.csv",header=None,engine="python")
NB_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_Tokenizer_50\Measures.csv",header=None,engine="python")
NB_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_Tokenizer_100\Measures.csv",header=None,engine="python")
NB_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_Tokenizer_1000\Measures.csv",header=None,engine="python")
NB_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_Tokenizer_10000\Measures.csv",header=None,engine="python")
NB_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_Tokenizer_20000\Measures.csv",header=None,engine="python")


LR = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_Tokenizer_All\Measures.csv",header=None,engine="python")
LR_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_Tokenizer_50\Measures.csv",header=None,engine="python")
LR_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_Tokenizer_100\Measures.csv",header=None,engine="python")
LR_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_Tokenizer_1000\Measures.csv",header=None,engine="python")
LR_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_Tokenizer_10000\Measures.csv",header=None,engine="python")
LR_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_Tokenizer_20000\Measures.csv",header=None,engine="python")


RF = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_Tokenizer_All\Measures.csv",header=None,engine="python")
RF_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_Tokenizer_50\Measures.csv",header=None,engine="python")
RF_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_Tokenizer_100\Measures.csv",header=None,engine="python")
RF_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_Tokenizer_1000\Measures.csv",header=None,engine="python")
RF_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_Tokenizer_10000\Measures.csv",header=None,engine="python")
RF_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_Tokenizer_20000\Measures.csv",header=None,engine="python")



#------------------------Classifier Accuracy Plot-----
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,NB.iloc[4][1]],[NB_50.iloc[2][1],NB_100.iloc[2][1],NB_1000.iloc[2][1],NB_10000.iloc[2][1],NB_20000.iloc[2][1],NB.iloc[2][1]],linewidth=8, label='Naive Bayes')
plt.plot([50,100,1000,10000,20000,LR.iloc[4][1]],[LR_50.iloc[2][1],LR_100.iloc[2][1],LR_1000.iloc[2][1],LR_10000.iloc[2][1],LR_20000.iloc[2][1],LR.iloc[2][1]], label='Logistic Reression')
plt.plot([50,100,1000,10000,20000,RF.iloc[4][1]],[RF_50.iloc[2][1],RF_100.iloc[2][1],RF_1000.iloc[2][1],RF_10000.iloc[2][1],RF_20000.iloc[2][1],RF.iloc[2][1]],color='red', label='Random Forest')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title('Tfidf Tokenizer')
plt.legend()
plt.savefig('Tfidf Tokenizer Accuracy',bbox_inches='tight')



#----------------------------------------------------------------------------------------------------
#---------------------Tfidf LemmaTokenizer------------------------
NB = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_LemmaTokenizer_All\Measures.csv",header=None,engine="python")
NB_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_LemmaTokenizer_50\Measures.csv",header=None,engine="python")
NB_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_LemmaTokenizer_100\Measures.csv",header=None,engine="python")
NB_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_LemmaTokenizer_1000\Measures.csv",header=None,engine="python")
NB_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_LemmaTokenizer_10000\Measures.csv",header=None,engine="python")
NB_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\Tfidf_LemmaTokenizer_20000\Measures.csv",header=None,engine="python")


LR = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_LemmaTokenizer_All\Measures.csv",header=None,engine="python")
LR_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_LemmaTokenizer_50\Measures.csv",header=None,engine="python")
LR_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_LemmaTokenizer_100\Measures.csv",header=None,engine="python")
LR_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_LemmaTokenizer_1000\Measures.csv",header=None,engine="python")
LR_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_LemmaTokenizer_10000\Measures.csv",header=None,engine="python")
LR_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\Tfidf_LemmaTokenizer_20000\Measures.csv",header=None,engine="python")


RF = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_LemmaTokenizer_All\Measures.csv",header=None,engine="python")
RF_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_LemmaTokenizer_50\Measures.csv",header=None,engine="python")
RF_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_LemmaTokenizer_100\Measures.csv",header=None,engine="python")
RF_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_LemmaTokenizer_1000\Measures.csv",header=None,engine="python")
RF_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_LemmaTokenizer_10000\Measures.csv",header=None,engine="python")
RF_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\Tfidf_LemmaTokenizer_20000\Measures.csv",header=None,engine="python")


print(NB_50.iloc[2][1])
#------------------------Classifier Accuracy Plot-----
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
plt.plot([50,100,1000,10000,20000,NB.iloc[4][1]],[NB_50.iloc[2][1],NB_100.iloc[2][1],NB_1000.iloc[2][1],NB_10000.iloc[2][1],NB_20000.iloc[2][1],NB.iloc[2][1]],linewidth=8, label='Naive Bayes')
plt.plot([50,100,1000,10000,20000,LR.iloc[4][1]],[LR_50.iloc[2][1],LR_100.iloc[2][1],LR_1000.iloc[2][1],LR_10000.iloc[2][1],LR_20000.iloc[2][1],LR.iloc[2][1]], label='Logistic Reression')
plt.plot([50,100,1000,10000,20000,RF.iloc[4][1]],[RF_50.iloc[2][1],RF_100.iloc[2][1],RF_1000.iloc[2][1],RF_10000.iloc[2][1],RF_20000.iloc[2][1],RF.iloc[2][1]],color='red', label='Random Forest')
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title('Tfidf LemmaTokenizer')
plt.legend()
plt.savefig('Tfidf LemmaTokenizer Accuracy',bbox_inches='tight')

