import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_indice_list(indices, bar_number, bar_width, spacing_ratio=0):

    # "Center" the bar number around 0, not clear but if you have 3 bar, 
    # bar_number_indices = [-1, 0, 1]
    bar_number_indices = [i - int(bar_number/2) for i in range(bar_number)]

    indices_list = []
    for number in bar_number_indices:

        indices_list.append([ ( ((number* bar_width) + (spacing_ratio*bar_width) * number) + ind) for ind in indices])

    return indices_list

total = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_All\Measures.csv",header=None,engine="python")

NB = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_All\Classification Report.csv",header=None,engine="python")
NB_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_50\Classification Report.csv",header=None,engine="python")
NB_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_100\Classification Report.csv",header=None,engine="python")
NB_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_1000\Classification Report.csv",header=None,engine="python")
NB_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_10000\Classification Report.csv",header=None,engine="python")
NB_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\CountVectorizer_Tokenizer_20000\Classification Report.csv",header=None,engine="python")


LR = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\CountVectorizer_Tokenizer_All\Classification Report.csv",header=None,engine="python")
LR_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\CountVectorizer_Tokenizer_50\Classification Report.csv",header=None,engine="python")
LR_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\CountVectorizer_Tokenizer_100\Classification Report.csv",header=None,engine="python")
LR_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\CountVectorizer_Tokenizer_1000\Classification Report.csv",header=None,engine="python")
LR_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\CountVectorizer_Tokenizer_10000\Classification Report.csv",header=None,engine="python")
LR_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\CountVectorizer_Tokenizer_20000\Classification Report.csv",header=None,engine="python")


RF = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\CountVectorizer_Tokenizer_All\Classification Report.csv",header=None,engine="python")
RF_50 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\CountVectorizer_Tokenizer_50\Classification Report.csv",header=None,engine="python")
RF_100 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\CountVectorizer_Tokenizer_100\Classification Report.csv",header=None,engine="python")
RF_1000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\CountVectorizer_Tokenizer_1000\Classification Report.csv",header=None,engine="python")
RF_10000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\CountVectorizer_Tokenizer_10000\Classification Report.csv",header=None,engine="python")
RF_20000 = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\CountVectorizer_Tokenizer_20000\Classification Report.csv",header=None,engine="python")



p = 2
n = 1
f = 0

print(NB_50)
print(float(NB_50.iloc[n][f]))

df = pd.DataFrame(dict(
    A=[float(NB_50.iloc[p][f]),float( NB_100.iloc[p][f]),float(NB_1000.iloc[p][f]),float(NB_10000.iloc[p][f]),float(NB_20000.iloc[p][f]),float(NB.iloc[p][f])],
    B=[float(NB_50.iloc[n][f]),float(NB_100.iloc[n][f]), float(NB_1000.iloc[n][f]),float(NB_10000.iloc[n][f]),float(NB_20000.iloc[n][f]),float( NB.iloc[n][f])],
    C=[float(LR_50.iloc[p][f]), float(LR_100.iloc[p][f]),float(LR_1000.iloc[p][f]),float(LR_10000.iloc[p][f]),float(LR_20000.iloc[p][f]),float(LR.iloc[p][f])],
    D=[float(LR_50.iloc[n][f]),float(LR_100.iloc[n][f]),float(LR_1000.iloc[n][f]),float(LR_10000.iloc[n][f]),float(LR_20000.iloc[n][f]),float(LR.iloc[n][f])],
    E=[float(RF_50.iloc[p][f]), float(RF_100.iloc[p][f]),float(RF_1000.iloc[p][f]),float(RF_10000.iloc[p][f]),float(RF_20000.iloc[p][f]),float(RF.iloc[p][f])],
    F=[float(RF_50.iloc[n][f]),float(RF_100.iloc[n][f]),float(RF_1000.iloc[n][f]),float(RF_10000.iloc[n][f]),float(RF_20000.iloc[n][f]),float(RF.iloc[n][f])]
))



plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')

NB_bar_list = [plt.bar(['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))], height = df.B, align='edge', width= -0.4,bottom = df.A, label = 'Negative NB'),
               plt.bar(['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))], height = df.A, align='edge', width= -0.4, label = 'Positive NB')]

LR_bar_list = [plt.bar(['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))], height = df.D, align='edge',bottom = df.C, width= 0.4,label = 'Negative LR' ),
               plt.bar(['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))], height = df.C, align='edge',width= 0.4, label = 'Positive LR')]

RF_bar_list = [plt.bar(['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))], height = df.F, align='center',bottom = df.E, width= 0.25, label = 'Negative RF' ),
               plt.bar(['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))], height = df.E, align='center',width= 0.25, label = 'Positive RF')]


#plt.xlim([-1,4])
plt.ylabel("F1 Score")
plt.xlabel("Number of Features")
plt.title('Tfidf LemmaTokenizer')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Tfidf LemmaTokenizer F1 Score plot',bbox_inches='tight')
plt.show()

plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')

 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [float(NB_50.iloc[p][f]),float( NB_100.iloc[p][f]),float(NB_1000.iloc[p][f]),float(NB_10000.iloc[p][f]),float(NB_20000.iloc[p][f]),float(NB.iloc[p][f])]
bars2 = [float(LR_50.iloc[p][f]), float(LR_100.iloc[p][f]),float(LR_1000.iloc[p][f]),float(LR_10000.iloc[p][f]),float(LR_20000.iloc[p][f]),float(LR.iloc[p][f])]
bars3 = [float(RF_50.iloc[p][f]), float(RF_100.iloc[p][f]),float(RF_1000.iloc[p][f]),float(RF_10000.iloc[p][f]),float(RF_20000.iloc[p][f]),float(RF.iloc[p][f])]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#4d88ff', width=barWidth, edgecolor='white', label='Naive Bayes')
plt.bar(r2, bars2, color='#ff751a', width=barWidth, edgecolor='white', label='Logistic Regression')
plt.bar(r3, bars3, color='#40bf80', width=barWidth, edgecolor='white', label='Random Forest')
 
# Add xticks on the middle of the group bars
plt.xlabel('Number of Features', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['50', '100', '1000', '10000','20000',str(int(total.iloc[4][1]))])
plt.ylabel("F1-score")
plt.title('CountVectorizer Tokenizer')
# Create legend & Show graphic
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('CountVectorizer Tokenizer Positive F1 Score plot',bbox_inches='tight')
plt.show()

plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')

 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [float(NB_50.iloc[n][f]), float(NB_100.iloc[n][f]),float(NB_1000.iloc[n][f]),float(NB_10000.iloc[n][f]),float(NB_20000.iloc[n][f]),float(NB.iloc[n][f])]
bars2 = [float(LR_50.iloc[n][f]), float(LR_100.iloc[n][f]),float(LR_1000.iloc[n][f]),float(LR_10000.iloc[n][f]),float(LR_20000.iloc[n][f]),float(LR.iloc[n][f])]
bars3 = [float(RF_50.iloc[n][f]), float(RF_100.iloc[n][f]),float(RF_1000.iloc[n][f]),float(RF_10000.iloc[n][f]),float(RF_20000.iloc[n][f]),float(RF.iloc[n][f])]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#4d88ff', width=barWidth, edgecolor='white', label='Naive Bayes')
plt.bar(r2, bars2, color='#ff751a', width=barWidth, edgecolor='white', label='Logistic Regression')
plt.bar(r3, bars3, color='#40bf80', width=barWidth, edgecolor='white', label='Random Forest')
 
# Add xticks on the middle of the group bars
plt.xlabel('Number of Features', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['50', '100', '1000', '10000','20000',str(int(total.iloc[4][1]))])
plt.ylabel("F1-score")
plt.title('CountVectorizer Tokenizer')
# Create legend & Show graphic
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('CountVectorizer Tokenizer Negative F1 Score plot',bbox_inches='tight')
plt.show()