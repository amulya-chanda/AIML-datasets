import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import contractions
from num2words import num2words
from textblob import TextBlob
# from spellchecker import SpellChecker
from gensim.models import Word2Vec
import re
import pickle
# from bs4 import BeautifulSoup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_excel('/users/amulya/Downloads/text_columns.xlsx')
df = df.fillna(0)

train, test = train_test_split(df,test_size=0.2, random_state=42)
print("Training set size:", len(train))
print("Testing set size:", len(test))
print("set size:", len(df))
company_backup_test = test['Company'].copy()

def pre_process_1(df):
    # icp_train_backup = df['ICP'].copy()
    company_train_backup = df['Company'].copy()
    df_dummies = pd.DataFrame({'Employees': df['Employees']})
    # Create binary columns based on employee categories
    df_dummies['emp_lessthan50'] = np.where(df_dummies['Employees'] <= 50, 1, 0)
    df_dummies['emp_50to100'] = np.where((df_dummies['Employees'] > 50) & (df_dummies['Employees'] <= 100), 1, 0)
    df_dummies['emp_100_above'] = np.where(df_dummies['Employees'] > 100, 1, 0)
    df = pd.concat([df, df_dummies], axis=1)  
    df['SEO Description'] = df['SEO Description'].replace(0, 'Others')
    df['SEO Description'] = df['SEO Description'].astype(str)
    df['SEO Description'] = df['SEO Description'].str.lower()
    
    df['SEO Description'] = df['SEO Description'].apply(word_tokenize)
    punctuation = set(string.punctuation)
    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [token for token in tokens if token not in punctuation])

    stop_words = set(stopwords.words('english'))
    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [token for token in tokens if token not in stop_words])

    lemmatizer = WordNetLemmatizer()
    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [num2words(token) if token.isdigit() else token for token in tokens])

    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [contractions.fix(token) for token in tokens])

    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [token for token in tokens if token.isalnum()])

    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [token.strip() for token in tokens])

    df['SEO Description'] = df['SEO Description'].apply(lambda tokens: [re.sub(r'http\S+', '', token) for token in tokens])

    df['SEO Description'] = df['SEO Description'].apply(' '.join)
    return df
# train = pre_process_1(train)
def seo_tfidf(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import MultiLabelBinarizer
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit(df['SEO Description'])
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    ########################
    # indLabels = LabelBinarizer()
    # df['Industry'] = df['Industry'].fillna('')
    # indLabels.fit(df['Industry'])
    # with open('industry.pkl', 'wb') as f:
    #     pickle.dump(indLabels, f)
    ########################
    lb_cities = LabelBinarizer()
    df['Company City'] = df['Company City'].astype(str)
    lb_cities.fit(df['Company City'])
    with open('lb_cities.pkl', 'wb') as f:
        pickle.dump(lb_cities, f) 
    ########################
    df['Technologies'] = df['Technologies'].replace(0, 'Other_technology')
    technologies_lists = df['Technologies'].str.split(',')
    mlb = MultiLabelBinarizer()
    technologies_encoded = mlb.fit(technologies_lists)  
    with open('technologies_encoded.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    ########################
    df['Keywords'] = df['Keywords'].replace(0, 'Other_keyword')
    keywords_lists = df['Keywords'].str.split(',')
    mlb = MultiLabelBinarizer()
    keywords_encoded = mlb.fit(keywords_lists)  
    with open('keywords_encoded.pkl', 'wb') as f:
        pickle.dump(mlb, f) 
    return df

# df = seo_tfidf(df)
# df = seo_process_pickel(df)
def seo_process_pickel(df):
    import pickle
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        loaded_tfidf_vectorizer = pickle.load(f)
    tfidf_matrix = loaded_tfidf_vectorizer.transform(df['SEO Description'])
    feature_names_test = loaded_tfidf_vectorizer.get_feature_names_out()
    seo_done_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names_test)
    df.reset_index(drop=True, inplace=True)
    seo_done_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, seo_done_df], axis=1)
    ########################
    # with open('industry.pkl', 'rb') as f:
    #     industryPickeler = pickle.load(f)
        
    # df['Industry'] = df['Industry'].fillna('')
    # industryMatrix = industryPickeler.transform(df['Industry'])
    # industryColumns = industryPickeler.classes_
    # industryDummies = pd.DataFrame(industryMatrix, columns=industryColumns)
    # industryDummies.reset_index(drop=True, inplace=True)
    ########################
    with open('lb_cities.pkl', 'rb') as f:
        lb_cities = pickle.load(f)
    df['Company City'] = df['Company City'].astype(str)
    cities_proc = lb_cities.transform(df['Company City'])
    cities_df = pd.DataFrame(cities_proc, columns=lb_cities.classes_)
    df.reset_index(drop=True, inplace=True)
    cities_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, cities_df], axis=1)
    #########################
    with open('technologies_encoded.pkl', 'rb') as f:
        mlb = pickle.load(f)
    df['Technologies'] = df['Technologies'].astype(str)
    technologies_lists = df['Technologies'].str.split(',')
    technologies_encoded = mlb.transform(technologies_lists)
    technologies_encoded_df = pd.DataFrame(technologies_encoded, columns=mlb.classes_)
    df.reset_index(drop=True, inplace=True)
    technologies_encoded_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    technologies_encoded_df.reset_index(drop=True, inplace=True)    
    df = pd.concat([df, technologies_encoded_df], axis=1)  
    ###########################
    with open('keywords_encoded.pkl', 'rb') as f:
        mlb = pickle.load(f)
    df['Keywords'] = df['Keywords'].astype(str)
    keywords_lists = df['Keywords'].str.split(',')
    keywords_encoded = mlb.transform(keywords_lists)
    keywords_encoded_df = pd.DataFrame(keywords_encoded, columns=mlb.classes_)
    df.reset_index(drop=True, inplace=True)
    keywords_encoded_df.reset_index(drop=True, inplace=True)

    df.reset_index(drop=True, inplace=True)
    keywords_encoded_df.reset_index(drop=True, inplace=True)
    
    df = pd.concat([df, keywords_encoded_df], axis=1)  
    return df
def drop_columns(df):
    
    df = df.drop(['Company', 'Lists', 'Employees','Company State', 'Company Country', 'SEO Description',
                  'Latest Funding','Industry','Company City','Keywords','Technologies','Last Raised At','Short Description',
                  'Latest Funding Amount','Number of Retail Locations', 'Founded Year','Total Funding','Annual Revenue'], axis = 1)
    # df['ICP'] = df['ICP'].replace({'ICP - 2': 1, 'ICP-1': 1, 'ICP - 1': 1, 'Not an ICP' : 0})
    return df
def all_methods(df):
    train_new = pre_process_1(df)
    train_new = seo_tfidf(train_new)
    train_new = seo_process_pickel(train_new)
    train_new = drop_columns(train_new)
    return train_new
# train_final = all_methods(train)
def all_methods_test(df):
    test_new = pre_process_1(df)
    test_new = seo_process_pickel(test_new)
    test_new = drop_columns(test_new)
    return test_new
# test_final = all_methods_test(test)
def analysis(y,y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
    train_accuracy = accuracy_score(y, y_pred)
    train_precision = precision_score(y, y_pred)
    train_recall = recall_score(y, y_pred)
    train_f1 = f1_score(y, y_pred)
    print(f'accuracy : {train_accuracy}\nprecision:{train_precision}\nrecall:{train_recall}\nf1_score:{train_f1}')
    return
def svm(df):
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler
    global svm_model
    y = df['ICP']
    X = df.drop('ICP', axis=1)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X)
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y)
    y_pred_train = svm_model.predict(X_train_scaled)
    analysis(y, y_pred_train)
    return svm_model
# svm(train_final)
def svm_test(df):
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler
    y = df['ICP']
    X = df.drop('ICP', axis=1)
    # Scale the features
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X)
    
    # Predict the labels for test data
    y_pred_test = svm_model.predict(X_test_scaled)
    analysis(y, y_pred_test)
    return
# svm_test(test_final)

with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

