import matplotlib.pyplot as plt
import csv
import numpy as np
import seaborn as sns

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def Union(lst1, lst2): 
    final_list = lst1 + lst2 
    return final_list 


total = pd.read_csv(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_All\Measures.csv",header=None,engine="python")


#---------------------------------ALL
NB_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_All\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         NB_pos.append(row)

LR_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_All\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         LR_pos.append(row)

RF_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_All\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         RF_pos.append(row)

#-------------------------------50
NB_50_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_50\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         NB_50_pos.append(row)

LR_50_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_50\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         LR_50_pos.append(row)
         
RF_50_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_50\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         RF_50_pos.append(row)  

#-------------------------------100         
NB_100_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_100\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         NB_100_pos.append(row)
 
LR_100_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_100\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         LR_100_pos.append(row)

RF_100_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_100\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         RF_100_pos.append(row)           

#-------------------------------1000  
NB_1000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_1000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         NB_1000_pos.append(row)         

LR_1000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_1000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         LR_1000_pos.append(row)

RF_1000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_1000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         RF_1000_pos.append(row)       

#-------------------------------10000  
NB_10000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_10000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         NB_10000_pos.append(row)         

LR_10000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_10000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         LR_10000_pos.append(row)

RF_10000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_10000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         RF_10000_pos.append(row)       

#-------------------------------20000  
NB_20000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Multinomial Naive Bayes\countVectorizer_LemmaTokenizer_20000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         NB_20000_pos.append(row)         

LR_20000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Logistic Regression\countVectorizer_LemmaTokenizer_20000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         LR_20000_pos.append(row)

RF_20000_pos = []
with open(r"C:\Users\Vicky\Desktop\Πτυχιακή\Undergaduate Thesis\Results\Random Forest\countVectorizer_LemmaTokenizer_20000\neg.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
         RF_20000_pos.append(row)       



NLR_50 = 0
NLR_100 = 0
NLR_1000 = 0
NLR_10000 = 0
NLR_20000 = 0

NLR = 0
NRR_50 = 0
NRR_100 = 0
NRR_1000 = 0
NRR_10000 = 0
NRR_20000 = 0

NRR = 0
LRR_50 =0 
LRR_100 = 0
LRR_1000 =0 
LRR_10000 = 0
LRR_20000 = 0

LRR = 0
#-----------------------JACCARD ALL        
NB_LR = len(Union(NB_pos,LR_pos))
NB_RF = len(Union(NB_pos,RF_pos))
LR_RF = len(Union(LR_pos,RF_pos))

NB_LR2 = len(intersection(NB_pos,LR_pos))
NB_RF2 = len(intersection(NB_pos,RF_pos))
LR_RF2 = len(intersection(RF_pos,LR_pos))


if (NB_LR!=0):
    NLR = NB_LR2/NB_LR
if(NB_RF!=0):
    NRR = NB_RF2/NB_RF
if (LR_RF!=0):
    LRR = LR_RF2/LR_RF
#print(NLR,NRR,LRR)

#-----------------------JACCARD 50
NB_LR_50 = len(Union(NB_50_pos,LR_50_pos))
NB_RF_50 = len(Union(NB_50_pos,RF_50_pos))
LR_RF_50 = len(Union(LR_50_pos,RF_50_pos))

NB_LR2_50 = len(intersection(NB_50_pos,LR_50_pos))
NB_RF2_50 = len(intersection(NB_50_pos,RF_50_pos))
LR_RF2_50 = len(intersection(RF_50_pos,LR_50_pos))


if (NB_LR_50!=0):
    NLR_50 = NB_LR2_50/NB_LR_50
if (NB_RF_50!=0):
    NRR_50 = NB_RF2_50/NB_RF_50
if (LR_RF_50!=0):
    LRR_50 = LR_RF2_50/LR_RF_50    
#print(NLR_50,NRR_50,LRR_50)
#-----------------------JACCARD 100
NB_LR_100 = len(Union(NB_100_pos,LR_100_pos))
NB_RF_100 = len(Union(NB_100_pos,RF_100_pos))
LR_RF_100 = len(Union(LR_100_pos,RF_100_pos))

NB_LR2_100 = len(intersection(NB_100_pos,LR_100_pos))
NB_RF2_100 = len(intersection(NB_100_pos,RF_100_pos))
LR_RF2_100 = len(intersection(RF_100_pos,LR_100_pos))


if (NB_LR_100!=0):
    NLR_100 = NB_LR2_100/NB_LR_100
if(NB_RF_100!=0):
    NRR_100 = NB_RF2_100/NB_RF_100
if (LR_RF_100!=0):
    LRR_100 = LR_RF2_100/LR_RF_100  
#print(NLR_100,NRR_100,LRR_100)
#-----------------------JACCARD 1000
NB_LR_1000 = len(Union(NB_1000_pos,LR_1000_pos))
NB_RF_1000 = len(Union(NB_1000_pos,RF_1000_pos))
LR_RF_1000 = len(Union(LR_1000_pos,RF_1000_pos))

NB_LR2_1000 = len(intersection(NB_1000_pos,LR_1000_pos))
NB_RF2_1000 = len(intersection(NB_1000_pos,RF_1000_pos))
LR_RF2_1000 = len(intersection(RF_1000_pos,LR_1000_pos))


if (NB_LR_1000!=0):
    NLR_1000 = NB_LR2_1000/NB_LR_1000
if(NB_RF_1000!=0):
    NRR_1000 = NB_RF2_1000/NB_RF_1000
if (LR_RF_1000!=0):
    LRR_1000 = LR_RF2_1000/LR_RF_1000  
#print(NLR_1000,NRR_1000,LRR_1000)

#-----------------------JACCARD 10000
NB_LR_10000 = len(Union(NB_10000_pos,LR_10000_pos))
NB_RF_10000 = len(Union(NB_10000_pos,RF_10000_pos))
LR_RF_10000 = len(Union(LR_10000_pos,RF_10000_pos))

NB_LR2_10000 = len(intersection(NB_10000_pos,LR_10000_pos))
NB_RF2_10000 = len(intersection(NB_10000_pos,RF_10000_pos))
LR_RF2_10000 = len(intersection(RF_10000_pos,LR_10000_pos))


if (NB_LR_10000!=0):
    NLR_10000 = NB_LR2_10000/NB_LR_10000
if(NB_RF_10000!=0):
    NRR_10000 = NB_RF2_10000/NB_RF_10000
if (LR_RF_10000!=0):
    LRR_10000 = LR_RF2_10000/LR_RF_10000  
#print(NLR_10000,NRR_10000,LRR_10000)

#-----------------------JACCARD 20000
NB_LR_20000 = len(Union(NB_20000_pos,LR_20000_pos))
NB_RF_20000 = len(Union(NB_20000_pos,RF_20000_pos))
LR_RF_20000 = len(Union(LR_20000_pos,RF_20000_pos))

NB_LR2_20000 = len(intersection(NB_20000_pos,LR_20000_pos))
NB_RF2_20000 = len(intersection(NB_20000_pos,RF_20000_pos))
LR_RF2_20000 = len(intersection(RF_20000_pos,LR_20000_pos))


if (NB_LR_20000!=0):
    NLR_20000 = NB_LR2_20000/NB_LR_20000
if(NB_RF_1000!=0):
    NRR_20000 = NB_RF2_20000/NB_RF_20000
if (LR_RF_20000!=0):
    LRR_20000 = LR_RF2_20000/LR_RF_20000  
#print(NLR_20000,NRR_20000,LRR_20000)




#-------------------------------Bar plot-------------------
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')

 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [NLR_50, NLR_100,NLR_1000,NLR_10000,NLR_20000,NLR]
bars2 = [NRR_50, NRR_100,NRR_1000,NRR_10000,NRR_20000,NRR]
bars3 = [LRR_50, LRR_100,LRR_1000,LRR_10000,LRR_20000,LRR]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#b300b3', width=barWidth, edgecolor='white', label='Naive Bayes - Logistic Regression')
plt.bar(r2, bars2, color='#1a53ff', width=barWidth, edgecolor='white', label='Random Forest - Naive Bayes')
plt.bar(r3, bars3, color='#009900', width=barWidth, edgecolor='white', label='Logistic Regression - Random Forest')
 
# Add xticks on the middle of the group bars
plt.xlabel('Number of Features', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['50', '100', '1000','10000','20000', str(int(total.iloc[4][1]))])
plt.ylabel("Jaccard Similarity")
plt.title('countVectorizer LemmaTokenizer')
# Create legend & Show graphic
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('countVectorizer LemmaTokenizer Jaccard Bar plot',bbox_inches='tight')
plt.show()
