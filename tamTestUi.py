from tamUi import *
import pandas as pd

import pickle
with open('svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)
def test_data_svm(df):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
    
    # test_backup_final = test_final['ICP'].copy()
    df.columns = df.columns.astype(str)
    # y = test_final['ICP']
    x = df
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(x)
    y_pred_test = svm.predict(X_scaled)
    print(y_pred_test)
    return y_pred_test

import streamlit as st
st.title("Upload TAM data")
# Upload file
uploaded_file = st.file_uploader("Upload xlsx file", type=['xlsx'])

if uploaded_file is not None:
    st.subheader('Original Data')
    df1 = pd.read_excel(uploaded_file)
    st.write(df.shape)
    st.write(df1.shape)
    df1.columns = df1.columns.astype(str)
    st.write(df1)
    df1 = df1.fillna(0)
    # Preprocess data
    test_final = all_methods_test(df1)
    processed_data = test_data_svm(test_final)
    processed_data = pd.DataFrame(processed_data)
    # df['predicted_icp'] = processed_data
    st.subheader('Processed Data')
    # new_data = st.write(test_final)
    
    # Allow user to download processed data
    st.download_button(
        label="Download predicted ICP",
        data=processed_data.to_csv().encode(),
        file_name='test_final.csv',
        mime='text/csv'
    )
