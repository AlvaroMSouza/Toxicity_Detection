import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import re
from bs4 import BeautifulSoup


from sklearn.metrics import auc, roc_curve, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# Load the data
train_df = pd.read_csv('train.csv', index_col='id', engine='python')
print(train_df.head())
print(train_df.info())

test_df = pd.read_csv('test.csv', index_col='id', engine='python')
print(test_df.head())

# Check for missing values
print(train_df.isnull().sum())

train_df = train_df.dropna(subset=['comment_text'])

print(train_df['comment_text'].isnull().sum())

print("Train and test shape: {} {}".format(train_df.shape, test_df.shape))


# Exploratory Data Analysis

# Check for class imbalance (<0.5 is non-toxic, >=0.5 is toxic)

train_df['toxicity'] = train_df['target'].apply(lambda x: 'Toxic' if x > 0.5 else 'Non-Toxic')

toxicity_counts = train_df['toxicity'].value_counts()

total_comments = len(train_df)
percentages = (toxicity_counts / total_comments) * 100

plt.figure(figsize=(8, 6))
toxicity_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Toxic and Non-Toxic Comments')
plt.xlabel('Toxicity')
plt.ylabel('Number of Comments')
plt.xticks(rotation=0)

for i, (count, percentage) in enumerate(zip(toxicity_counts, percentages)):
    plt.text(i, count + 0.1, f'{percentage:.1f}%', ha='center', va='bottom')

plt.show()

# RESULT: DATASET IS IMBALANCE!!



toxic_df = train_df[train_df['target'] > 0.5]


# Analyze Toxicity Subtype Features

features_subtype = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

def plot_toxicity_based_on_features(toxic_df, title, features):
    toxic_counts = toxic_df[features].apply(lambda x: (x > 0.5).sum())

    total_toxic_comments = len(toxic_df)
    percentages = (toxic_counts / total_toxic_comments) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x=percentages.index, y=percentages.values, palette='viridis')
    plt.title(title)
    plt.ylabel('Percentage of Toxic Comments')
    plt.ylim(0, 100)  

    for i, percentage in enumerate(percentages):
        plt.text(i, percentage + 1, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.show()

plot_toxicity_based_on_features(toxic_df,'Percentage of toxicity nature in toxic comments data', features_subtype)

# RESULT: MOST TOXIC COMMENTS ARE INSULTS!!


# Preprocessing

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    return text


train_df['preprocessed_text'] = train_df['comment_text'].apply(clean_text)  
test_df['preprocessed_text'] = test_df['comment_text'].apply(clean_text)

print(train_df['preprocessed_text'].head())

# Tokenization
train_df['preprocessed_text'] = train_df['preprocessed_text'].apply(word_tokenize)

# Remove stopwords and special characters
stop_words = set(stopwords.words('english'))

train_df['preprocessed_text'] = train_df['preprocessed_text'].apply(lambda x: [word for word in x if word not in stop_words])

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    lemm_text = [lemmatizer.lemmatize(word) for word in text] 
    return lemm_text

train_df['preprocessed_text'] = train_df['preprocessed_text'].apply(lambda x:lemmatizer(x))

train_df['preprocessed_text'] = train_df['preprocessed_text'].apply(lambda x: ' '.join(x))

print(train_df[['comment_text', 'preprocessed_text']].head())


print(train_df['preprocessed_text'].head())

# Tokenization
test_df['preprocessed_text'] = test_df['preprocessed_text'].apply(word_tokenize)

# Remove stopwords and special characters
stop_words = set(stopwords.words('english'))

test_df['preprocessed_text'] = test_df['preprocessed_text'].apply(lambda x: [word for word in x if word not in stop_words])

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

test_df['preprocessed_text'] = test_df['preprocessed_text'].apply(lambda x:lemmatizer(x))

test_df['preprocessed_text'] = test_df['preprocessed_text'].apply(lambda x: ' '.join(x))

print(test_df[['comment_text', 'preprocessed_text']].head())



features = train_df['preprocessed_text']
target = train_df['target']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

X_test = test_df['preprocessed_text']

X_train.to_pickle('Split_Dataset/X_train.pkl')
X_val.to_pickle('Split_Dataset/X_cv.pkl')
X_test.to_pickle('Split_Dataset/X_test.pkl')
y_train.to_pickle('Split_Dataset/y_train.pkl')
y_val.to_pickle('Split_Dataset/y_cv.pkl')


count_vectorizer = CountVectorizer(ngram_range=(1,2), max_features=30000)
count_train  = count_vectorizer.fit_transform(X_train)
count_val  = count_vectorizer.transform(X_val)
count_test  = count_vectorizer.transform(X_test)

print(count_train.shape, count_val.shape, count_test.shape)

# Train the LinearRegression model

lr = LinearRegression()
lr.fit(count_train, y_train)

y_pred_train = lr.predict(count_train)
err_lr = mean_squared_error(y_train, y_pred_train)
print('Mean Squared Error Training:', err_lr)

y_pred_val = lr.predict(count_val)
err_lr_val = mean_squared_error(y_val, y_pred_val) 
print('Mean Squared Error Validation:', err_lr_val)

# Train the XGBRegressor model

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(count_train, y_train)

y_pred_train = xgb.predict(count_train)
err_xgb = mean_squared_error(y_train, y_pred_train)
print('Mean Squared Error Training:', err_xgb)

y_pred_val = xgb.predict(count_val)
err_xgb_val = mean_squared_error(y_val, y_pred_val)
print('Mean Squared Error Validation:', err_xgb_val)

