#!/usr/bin/env python
# coding: utf-8

# # YELP REVIEW CLASSIFICATION

# ### IMPORT DATA AND LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


yelp_df = pd.read_csv('yelp.csv')
yelp_df.head(7)


# ### DATA VISUALIZATION

# In[3]:


yelp_df.describe()


# In[4]:


### Adding length of all the text in a cloumn
yelp_df['length'] = yelp_df['text'].apply(len)
yelp_df.head()


# In[5]:


yelp_df.length.plot(bins=100, kind='hist')
plt.show()


# In[6]:


yelp_df.length.describe()


# In[7]:


## LONGEST MESSAGE AND SHORTEST MESSAGE
max_val = yelp_df.length.describe()['max']
min_val = yelp_df.length.describe()['min']
mean_val = int(yelp_df.length.describe()['50%'])

text_max = yelp_df[yelp_df['length'] == max_val]['text'].iloc[0]
text_min = yelp_df[yelp_df['length'] == min_val]['text'].iloc[0]
text_mean = yelp_df[yelp_df['length'] == mean_val]['text'].iloc[0]


print('MAX TEXT : ', text_max)
print('\n#########################  ############################\n')
print('MIN TEXT : ', text_min)
print('\n#########################  ############################\n')
print('MEAN TEXT : ', text_mean)


# In[8]:


## COUNT OF STARS
plt.figure(figsize=(10, 7))
sns.countplot(y='stars', data=yelp_df)
plt.xlabel('Count Of Stars', fontsize=17)
plt.show()


# In[9]:


### FACETGRID OF LENGTH AND STARS
g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5)
g.map(plt.hist, 'length', bins=20, color='r', ec='black')
plt.show()


# In[10]:


## REVIEWS OF 1 and 5 stars

yelp_df_1 = yelp_df[yelp_df['stars'] == 1]
yelp_df_5 = yelp_df[yelp_df['stars'] == 5]

yelp_df_1_5 = pd.concat([yelp_df_1, yelp_df_5])

yelp_df_1_5.sample(7)


# In[11]:


yelp_df_1_5.info()


# In[12]:


### PERCENTAGES OF 1 and 5 Stars

star1percnt = 100*len(yelp_df_1)/len(yelp_df_1_5)
star5percnt = 100*len(yelp_df_5)/len(yelp_df_1_5)

print('1-Star Percentage : {}%'.format(round(star1percnt, 2)))
print('5-Star Percentage : {}%'.format(round(star5percnt, 2)))

sns.countplot(yelp_df_1_5['stars'])
plt.show()


# ### CREATING TEST AND TRAIN DATA

# #### REMOVE PUNCTUATION

# In[13]:


import string
string .punctuation


# In[14]:


test = 'hello #Mr.Future, I am too [happy] to learn AI!'

test_punct_removed = [char for char in test if char not in string.punctuation]
test_punct_removed


# In[15]:


test_punct_removed = ''.join(test_punct_removed)
test_punct_removed


# #### REMOVE STOPWORDS

# In[17]:


import nltk


# In[18]:


nltk.download('stopwords')


# In[20]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[21]:


test_punct_removed


# In[22]:


stopwords_removed = [word for word in test_punct_removed.split() if word.lower() not in stopwords.words('english')]
stopwords_removed


# ### MESSAGE CLEANER FUNCTION

# In[23]:


def message_clean(message):
    punct_removed = [char for char in message if char not in string.punctuation]
    punct_removed_join = ''.join(punct_removed)
    stopwords_removed = [word for word in punct_removed_join.split() if word.lower() not in stopwords.words('english')]
    return stopwords_removed


# In[24]:


yelp_df_clean = yelp_df_1_5['text'].apply(message_clean)
yelp_df_clean.head()


# In[25]:


print(yelp_df_clean[0]) # Cleaned Message


# In[26]:


print(yelp_df_1_5['text'][0]) # Original Message


# ### COUNT VECTORIZER

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer=message_clean)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])


# In[28]:


print(yelp_countvectorizer.toarray())


# In[29]:


yelp_countvectorizer.shape


# ### TRAINING MODEL WITH ALL DATASET

# In[30]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values


NB_classifier.fit(yelp_countvectorizer, label)


# In[31]:


test_sample = ['amazing food! highly recommended for everyone']

test_sample_cv = vectorizer.transform(test_sample)
test_predcit = NB_classifier.predict(test_sample_cv)

test_predcit


# In[32]:


test_sample = ['worst food i ever tasted']

test_sample_cv = vectorizer.transform(test_sample)
test_predcit = NB_classifier.predict(test_sample_cv)

test_predcit


# ### DIVIDING TRAIN TEST

# In[33]:


from sklearn.model_selection import train_test_split

X = yelp_countvectorizer
y = label

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=47)


# In[34]:


NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[35]:


# from sklearn.naive_bayes import GaussianNB
# GNB_classifier = GaussianNB()
# GNB_classifier.fit(X_train.toarray(), y_train)

# GNB_classifier.score(X_train.toarray(), y_train)

# GNB_classifier.score(X_test.toarray(), y_test)


# ### EVALAUATION OF MODELS

# In[36]:


NB_classifier.score(X_train, y_train)


# In[37]:


NB_classifier.score(X_test, y_test)


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix


# In[39]:


y_predict_train = NB_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True, cmap='ocean')
plt.show()


# In[40]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True, cmap='ocean')
plt.show()


# In[41]:


print(classification_report(y_test, y_predict_test))

