#!/usr/bin/env python
# coding: utf-8

# In[229]:


import numpy as np
import pandas as pd


# In[230]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[231]:


df.sample(5)


# In[232]:


df.shape


# In[233]:


# 1. Data Cleaning
# 2. EDA
# 3. Text Reprocessing
# 4. Model Building
# 5. Evaluation
# 6. Imrovement
# 7. Website
# 8. Deploy


# In[234]:


# 1. Data Cleaning


# In[235]:


df.info()


# In[236]:


# drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[237]:


df.sample(5)


# In[238]:


# Renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[239]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[240]:


df['target'] = encoder.fit_transform(df['target'])


# In[241]:


df.head()


# In[242]:


# Missing values
df.isnull().sum()


# In[243]:


# check for duplicate values
df.duplicated().sum()


# In[244]:


# Remove Duplicated Values
df = df.drop_duplicates(keep = 'first')


# In[245]:


df.duplicated().sum()


# In[246]:


df.shape


# In[247]:


# 2. EDA --> Exporatory Data Analysis


# In[248]:


df.head()


# In[249]:


df['target'].value_counts()


# In[250]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels = ['ham', 'spam'], autopct = "%0.2f")
plt.show()


# In[251]:


# Data is Imbalance


# In[252]:


import nltk


# In[253]:


nltk.download('punkt')


# In[254]:


df['num_characters'] = df['text'].apply(len)


# In[255]:


df.head()


# In[256]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[257]:


df.head()


# In[258]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[259]:


df.head()


# In[260]:


df[['num_characters', 'num_words', 'num_sentences']].describe()


# In[261]:


# ham messages
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()


# In[262]:


# spam messages
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()


# In[263]:


import seaborn as sns


# In[264]:


plt.figure(figsize = (12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color = 'red')


# In[265]:


sns.pairplot(df, hue = 'target')


# In[266]:


numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot = True)


# In[267]:


# 3. Data Processing
# - Lower Case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - stemming/Lemitization


# In[268]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    y = text[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english')  and i not in string.punctuation:
            y.append(i)
    
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[269]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem(('loving'))


# In[270]:


import nltk
nltk.download('stopwords')
from  nltk.corpus import stopwords
stopwords.words('english')


# In[271]:


import string 
string.punctuation


# In[272]:


transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')


# In[273]:


df['text'][0]


# In[274]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem(('loving'))


# In[275]:


df['transform_text'] = df['text'].apply(transform_text)


# In[276]:


from wordcloud import WordCloud
wc = WordCloud(width = 500, height = 400, min_font_size = 10,background_color = 'white')


# In[277]:


spam_wc = wc.generate(df[df['target'] == 1]['transform_text'].str.cat(sep = " "))


# In[278]:


plt.figure(figsize = (15,6))
plt.imshow(spam_wc)


# In[279]:


ham_wc = wc.generate(df[df['target'] == 0]['transform_text'].str.cat(sep = " "))


# In[280]:


plt.figure(figsize = (12,6))
plt.imshow(ham_wc)


# In[281]:


df.head()


# In[282]:


spam_corpus = []
for msg in df[df['target'] == 1]['transform_text'].tolist():
    for word in msg.split():
            spam_corpus.append(word)


# In[283]:


len(spam_corpus)


# In[284]:


from collections import Counter
word_counts = Counter(spam_corpus).most_common(30)
df_word_counts = pd.DataFrame(word_counts, columns=['0', '1'])

# Plotting using seaborn barplot
sns.barplot(x='0', y='1', data=df_word_counts)
plt.xticks(rotation = 'vertical')


# In[285]:


ham_corpus = []
for msg in df[df['target'] == 0]['transform_text'].tolist():
    for word in msg.split():
            ham_corpus.append(word)


# In[286]:


len(ham_corpus)


# In[287]:


from collections import Counter
word_counts = Counter(ham_corpus).most_common(30)
df_word_counts = pd.DataFrame(word_counts, columns=['0', '1'])

# Plotting using seaborn barplot
sns.barplot(x='0', y='1', data=df_word_counts)
plt.xticks(rotation = 'vertical')


# In[288]:


# Text Vectorization
# Using Bag Of Words
df.head()


# In[289]:


# 4. Model Building


# In[323]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 3000)


# In[324]:


X = tfidf.fit_transform(df['transform_text']).toarray()


# In[325]:


X.shape


# In[326]:


y = df['target'].values


# In[327]:


y


# In[328]:


# from sklearn.model_selection import train_test_split


# In[329]:


# X_train,X_test,y_train,y_Test = train_test_split(X,y,test_size = 0.2,random_state = 2)


# In[330]:


# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[331]:


# gnb = GaussianNB()
# mnb = MultinomialNB()
# bnb = BernoulliNB()


# In[332]:


# gnb.fit(X_train,y_train)
# y_pred1 = gnb.predict(X_test)
# print(accuracy_score(y_test,y_pred1))
# print(confusion_metrix(y_test,y_pred1))
# print(precision_score(y_test,y_pred1))


# In[333]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[334]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[335]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[336]:


gnb.fit(X_train, y_train)

# Predicting labels for the test set
y_pred1 = gnb.predict(X_test)


# In[337]:


# Evaluating the classifier
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))


# In[338]:


mnb.fit(X_train, y_train)

# Predicting labels for the test set
y_pred2 = mnb.predict(X_test)


# In[339]:


# Evaluating the classifier
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))


# In[340]:


bnb.fit(X_train, y_train)

# Predicting labels for the test set
y_pred3 = bnb.predict(X_test)


# In[341]:


# Evaluating the classifier
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("Precision:", precision_score(y_test, y_pred3))


# In[342]:


# tfidf --> MNB


# In[343]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[344]:


svc = SVC(kernel = 'sigmoid', gamma = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth = 5)
lrc = LogisticRegression(solver = 'liblinear', penalty = '11')
rfc = RandomForestClassifier(n_estimators = 50, random_state = 2)
abc = AdaBoostClassifier(n_estimators = 50, random_state = 2)
bc = BaggingClassifier(n_estimators = 50, random_state = 2)
etc = ExtraTreesClassifier(n_estimators = 50, random_state = 2)
gdbt = GradientBoostingClassifier(n_estimators = 50, random_state = 2)
xgb = XGBClassifier(n_estimators = 50, random_state = 2)


# In[364]:


from sklearn.linear_model import LogisticRegression
Clfs = {
    
    'SVC' : svc,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT' : GradientBoostingClassifier(),
    'xgb' : xgb
    
}


# In[365]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[366]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[367]:


# accuracy_scores = []
# precision_scores = []

# for name,clf in Clfs.items():
    
#     current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
#     print("For ",name)
#     print("Accuracy - ",current_accuracy)
#     print("Precision - ",current_precision)
    
#     accuracy_scores.append(current_accuracy)
#     precision_scores.append(current_precision)


# In[349]:


# performance_df = pd.DataFrame({'Algorithm':Clfs.keys(),'Accuracy':accuracy_scores,'precision':precision_scores}).sort_values('Precision',ascending = False)


# In[350]:


# Model improve 
# 1. Change the max_features parameter of Tfidf


# In[369]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier

# Initialize classifiers
Clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': LogisticRegression(),  # Initialize LogisticRegression without specifying penalty
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': GradientBoostingClassifier(),
    'xgb': xgb
}

# Define train_classifier function
def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

# Initialize lists to store scores
accuracy_scores = []
precision_scores = []

# Loop over classifiers and train
for name, clf in Clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print("For", name)
    print("Accuracy -", current_accuracy)
    print("Precision -", current_precision)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[371]:


performance_df = pd.DataFrame({'Algorithm':Clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[372]:


performance_df


# In[373]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[374]:


performance_df1


# In[375]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[376]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[378]:


temp_df = pd.DataFrame({'Algorithm':Clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[379]:


temp_df = pd.DataFrame({'Algorithm':Clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[380]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[381]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[382]:


temp_df = pd.DataFrame({'Algorithm':Clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[383]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[384]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[385]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[386]:


voting.fit(X_train,y_train)


# In[387]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[388]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[389]:


from sklearn.ensemble import StackingClassifier


# In[390]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[391]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[392]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




