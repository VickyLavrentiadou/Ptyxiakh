import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve 
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize 
from collections import defaultdict 
import csv
import random
from slacker import Slacker 


class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]




def NB_Important_Features(vectorizer, clf, n):
 feature_names = vectorizer.get_feature_names()
 features = sorted(zip(clf.coef_[0], feature_names),reverse=True)
 #print(features)
 top = zip(features[:n], features[:-(n + 1):-1])
 print("----Most informative features--------------Least Informative features-----------")
 print("\n")
 for (coef_1, fn_1), (coef_2, fn_2) in top:
    print( "\t%.4f\t%15s\t\t%.4f\t%15s" % (coef_1, fn_1, coef_2, fn_2))
 
    
 plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
 plt.bar(range(len(features[:n])), [val[0] for val in features[:n]], align='center', color=(0.34, 0.1, 0.86, 0.6))
 plt.xticks(range(len(features[:n])), [val[1] for val in features[:n]])
 plt.xticks(rotation=70)
 plt.savefig('Important Features',bbox_inches='tight')
 plt.show()

        
def Regression_Important_Features(classifier, feature_names, top_features=10):
 coef = classifier.coef_.ravel()
 #print(coef)
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 
 # create plot
 plt.figure(num=None, figsize=(8, 5), dpi=140, facecolor='w', edgecolor='k')
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients],align='center', color=colors)
 
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.savefig('Important Features',bbox_inches='tight')
 plt.show()

def Forest_Important_Features(vectorizer,clf,n):
  feature_names = vectorizer.get_feature_names()
  features = sorted(zip(clf.feature_importances_, feature_names),reverse=True)
  #print(coefs_with_fns[:n])
  top = zip(features[:n], features[:-(n + 1):-1])
  print("----Most informative features--------------Least Informative features-----------")
  print("\n")
  for (imp_1, feats_1), (imp_2, feats_2) in top:
     print( "\t%.4f\t%15s\t\t%.4f\t%15s" % (imp_1, feats_1, imp_2, feats_2))
  
  plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
  plt.bar(range(len(features[:n])), [val[0] for val in features[:n]], align='center', color=(0.4, 0.2, 0.6, 0.6))
  plt.xticks(range(len(features[:n])), [val[1] for val in features[:n]])
  plt.xticks(rotation=70)
  plt.savefig('Important Features',bbox_inches='tight')
  plt.show()

        



df = pd.read_csv("MyData.csv")


df['reviewText'] = df['reviewText'].str.replace('\d+', '')
df['reviewText'] = df['reviewText'].str.findall('\w{4,}').str.join(' ')

#----------------Count Vectorizer------------
token = RegexpTokenizer(r'[a-zA-Z]+')


#----------Count Vectorizer---------------------
#cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),max_features=50,tokenizer = token.tokenize)
#cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = LemmaTokenizer())


#----------Tf-Idf Vectorizer---------------------
#cv = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1), tokenizer = token.tokenize)
cv = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),max_features=50,tokenizer = LemmaTokenizer())



text_counts= cv.fit_transform(df['reviewText'].values.astype('U'))

#print(text_counts.toarray())
#print((cv.get_feature_names()))
feats = len(cv.get_feature_names())
#-----------------Splitting Dataset---------------
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df['Ratings'], test_size=0.3, random_state=1)

print(X_train)
#print(y_train)

#------------Classifiers------------------- 
clf = MultinomialNB().fit(X_train, y_train)
#clf =LogisticRegression().fit(X_train, y_train)
#clf = RandomForestClassifier(n_estimators=100).fit(X_train,y_train)

#----------------Predictions---------------
predicted = clf.predict(X_test)
print(len(predicted))


#---------------Most Frequent Positive and Negative words--------
test = X_test.toarray()

pos = defaultdict(list)
neg = defaultdict(list)
positive = []
for i in range(len(test)):
   for k,l in sorted(zip(cv.get_feature_names(),np.asarray(X_test[i].sum(axis=0)).ravel())):
       if(l!=0):
           if(predicted[i]==1):
               pos[k].append(l)
           elif(predicted[i]==0):
               neg[k].append(l)
 
 
positive = {}
for key,value in pos.items():
    positive[key] = sum(value)

negative = {}
for key,value in neg.items():
    negative[key] = sum(value)


delete_positive= []
delete_negative =[]

for (key_pos, value_pos) in positive.items():
    for (key_neg, value_neg) in negative.items():
        if(key_pos == key_neg ):
            if( value_pos > value_neg):
                delete_negative.append(key_neg)
               
            else: 
                delete_positive.append(key_pos)
 
               
for i in delete_negative:
    del negative[i]  

for j in delete_positive:
    del positive[j]


with open('pos.csv', 'w', newline="") as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in positive.items():
       writer.writerow([key])

with open('Positive frequencies.csv', 'w', newline="") as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in positive.items():
       writer.writerow([key, value])

with open('neg.csv', 'w', newline="") as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in negative.items():
       writer.writerow([key])

with open('Negative frequencies.csv', 'w', newline="") as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in negative.items():
       writer.writerow([key, value])
#print(sorted(negative.items(), key=lambda x:x[1]))
#most_freq_pos =  nlargest(3, positive, key=positive.get)
#print(most_freq_pos)

#Pos_words.write(positive[3])
#Neg_words.write(negative[3])

#-------------Important Features------------------
#Forest_Important_Features(cv,clf,20)
#Regression_Important_Features(clf,cv.get_feature_names(), top_features=10)
NB_Important_Features(cv, clf, 20)


#--------------Positive and Negative Predictions--------
pos = 0
neg = 0 
for i in predicted:
    if i==1:
        pos = pos + 1
    else:
        neg = neg + 1

#-------------Word Frequencies/Imbalance--------
sum_words = text_counts.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in     cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)



imbalance = pd.crosstab(index = df['Ratings'], columns="Total count")

print(imbalance)
print('\n')

#--------------------------Imbalance-------------------
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
df['Ratings'].value_counts().plot(kind='bar')
plt.title('Imbalance of Data')
plt.ylabel('Number of reviews')
plt.xlabel('Ratings')
plt.savefig('Imbalance of Data')
plt.show()


#------------------------------Ratings bar plot
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
df['overall'].value_counts().plot(kind='bar')
plt.title('Reviewers Ratings')
plt.ylabel('Number of reviews')
plt.xlabel('Ratings')
plt.savefig('Ratings')
plt.show()


#-------------------------Total ,train, test set
total_reviews = df.shape[0]
train_total_reviews = len(predicted)
test_total_reviews = total_reviews - train_total_reviews

plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')
height = [total_reviews,test_total_reviews,train_total_reviews]
bars = ('Total reviews', 'Training set total reviews', 'Test set total reviews')
y_pos = np.arange(len(bars))
 

plt.barh(y_pos, height, color=[ '#66cc99', '#d9b3ff', '#99bbff']) 
plt.yticks(y_pos, bars)
plt.savefig('Total reviews, Total training reviews and Total test reviews',bbox_inches='tight') 
plt.show()


#------------------------------------Predicted real positive and Negative reviews
tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
total = tn + fp + fn + tp
Tpos = fn + tp
Tneg = fp + tn

Tpos_perc = (Tpos/total)*100
Tneg_perc = (Tneg/total)*100

pred_total  = pos + neg
pos_perc = (pos/pred_total)*100
neg_perc = (neg/pred_total)*100

group_names=[str(pos)+"\n"+str(round(pos_perc))+"%", str(neg)+"\n"+str(round(neg_perc))+"%"]
group_size=[pos,neg]
subgroup_names=[str(Tpos)+"\n"+str(round(Tpos_perc))+"%",str(Tneg)+"\n"+str(round(Tneg_perc))+"%"]
subgroup_size=[Tpos,Tneg]
names = ['Predicted Positive','Predicted Negative','Real Positive','Real Negative']

                    
a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

accuracy = round(100*metrics.accuracy_score(y_test, predicted))
acc = str(accuracy)
patch = mpatches.Patch(color='#333333', label="Accuracy: "+acc+"%")
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.8, labels=group_names,textprops={'fontsize': 14}, colors=[a(0.6), b(0.6)] )
plt.setp( mypie, width=0.45, edgecolor='white')


 
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.8-0.3, labels=subgroup_names, 
                   labeldistance=0.7,textprops={'fontsize': 14}, colors=[a(0.4),b(0.4)])
plt.setp( mypie2, width=0.5, edgecolor='white')
plt.margins(0,0)
ax.legend(labels = names,loc=l, bbox_to_anchor=(0.9,0.6),prop={'size':15})
plt.legend(handles=[patch])
plt.savefig('Predicted-Real reviews',bbox_inches='tight')
plt.show()

#-----------------------------Precision-Recall
average_precision = average_precision_score(y_test, predicted)
avg_p = str(round(100*average_precision))

precision, recall, thresholds = precision_recall_curve(y_test, predicted)
plt.figure(num=None, figsize=(7, 4), dpi=120, facecolor='w', edgecolor='k')

patch = mpatches.Patch( label="Avg Precision-Recall score: "+avg_p+"%")
plt.plot(precision, recall, label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve')
plt.legend(handles=[patch])
plt.savefig('Precision-Recall',bbox_inches='tight')
plt.show()

#-----------------------Measure CSV creation
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
data = {'positive': pos, 'negative': neg, 'accuracy': accuracy, 'Average Precision': average_precision, 'Total Features': feats}
measures = pd.Series(data).transpose()

measures.to_csv('Measures.csv', index=True)

#---------------------------------------Confusion Matrix
cm = metrics.confusion_matrix(y_test, predicted)

plt.figure(num=None, figsize=(7, 4), dpi=140, facecolor='w', edgecolor='k')
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.nipy_spectral)
classNames = ['Negative','Positive']
plt.title('Confusion Matrix for Random Forest Predictions')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
       if(j==1 and i==1):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]),horizontalalignment="center", 
                 color="black" )
       else:
         plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]),horizontalalignment="center", 
                 color="white" )
plt.savefig('Confusion Matrix',bbox_inches='tight')
plt.show()

#--------------------------------------Classification report
classNames = ['Negative','Positive']
print(classification_report(y_test,predicted, target_names=classNames ))



clsf_report = pd.DataFrame(classification_report(y_true = y_test, y_pred = predicted, output_dict=True)).transpose()
clsf_report.to_csv('Classification Report.csv', index= False)

x = random.randint(1,101)
slack = Slacker('xoxb-852121624103-838910014995-g6SnfXkGC9iTE3UcRWsdbrrA')
message = "Vicky it is done..."+str(x)
slack.chat.post_message('#sentiment-analysis-amazon-reviews',message)

#------------------------------------word cloud ,most frequent words


mask = np.array(Image.open(r"C:\Users\Vicky\Desktop\Πτυχιακή\json data\cloud2.png"))

wc1 = WordCloud(width = 620, height = 620, background_color='white',relative_scaling=0.5,
               normalize_plurals=False, mask = mask).generate(str(words_freq))

plt.figure(figsize=(15,10),facecolor = 'white', edgecolor='none')
plt.imshow(wc1)
plt.axis('off')
plt.savefig('Frequent words')
plt.show()


#------------------Positive Vocabulary------------------
mask = np.array(Image.open(r"C:\Users\Vicky\Desktop\Πτυχιακή\json data\cloud2.png"))

wc2 = WordCloud(width = 620, height = 620, background_color='white',relative_scaling=0.5,
               normalize_plurals=False, mask = mask).generate_from_frequencies(positive)

plt.figure(figsize=(15,10),facecolor = 'white', edgecolor='none')
plt.imshow(wc2)
plt.axis('off')
plt.savefig('Positive Vocabulary')
plt.show()
#---------------Negative Vocabulary-------------------
mask = np.array(Image.open(r"C:\Users\Vicky\Desktop\Πτυχιακή\json data\cloud2.png"))

wc3 = WordCloud(width = 620, height = 620, background_color='white',relative_scaling=0.5,
               normalize_plurals=False, mask = mask).generate_from_frequencies(negative)

plt.figure(figsize=(15,10),facecolor = 'white', edgecolor='none')
plt.imshow(wc3)
plt.axis('off')
plt.savefig('Negative Vocabulary')
plt.show()





