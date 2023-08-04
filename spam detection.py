# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:51:30 2023

@author: Lenovo
"""
import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\Lenovo\\Downloads\\spam.csv",encoding='latin-1')

df.shape

df.head(5)
df.info()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
#renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.head(2)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
df.head(2)
df.isnull().sum()
df.duplicated().sum()

print("before removing duplicates;",df.shape)
df.drop_duplicates(keep='first',inplace=True)
print("after removing duplicates",df.shape)

#Checking counts of Ham and spam
df['target'].value_counts().plot(kind='bar')
import matplotlib.pyplot as plt
plt.pie(df["target"].value_counts(),labels=['ham','spam'],autopct='%1.1f%%')
plt.show()
import nltk
#num of characters
df['num_characters'] = df['text'].apply(len)

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head(2)
df[['num_characters','num_words','num_sentences']].describe()
# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()

# spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

#num_characters
import seaborn as sns
sns.histplot(df[df['target'] == 0]['num_characters'], label='Non-spam')
sns.histplot(df[df['target'] == 1]['num_characters'], label='Spam')

plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Characters')

plt.legend()

plt.show()


df.corr()
#visualize correlation
sns.heatmap(df.corr(),annot=True)
