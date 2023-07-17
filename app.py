import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk
import streamlit as st


uploaded_file = st.file_uploader('Upload a file', type="csv")
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df=df.head(100)
    
    st.header('Count of number of stars given')
    ax = df['Score'].value_counts().sort_index()
    st.bar_chart(ax)
    


    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)


    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2]
        }
        return scores_dict



    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Text']
            myid = row['Id']
            roberta_result = polarity_scores_roberta(text)
            
            res[myid] = roberta_result
        except RuntimeError:
            print(f'Broke for id {myid}')


    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'Id'})
    results_df = results_df.merge(df, how='left')


    def plotPieChart(df):
        roberta_neg_avg=np.mean(df['roberta_neg'])  
        roberta_pos_avg=np.mean(df['roberta_pos'])  
        roberta_neu_avg=np.mean(df['roberta_neu'])  

        data = [roberta_neg_avg,roberta_pos_avg,roberta_neu_avg]
        keys = ['Negative', 'Positive','Neutral']
       
    
        fig1, ax1 = plt.subplots()
        ax1.pie(data, labels=keys, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

    results_df_sorted=results_df.sort_values('Time')   
    st.header('Average sentiment count of product')
    plotPieChart(results_df_sorted)

    st.header('Most frequent comment in review')
    st.write(results_df['Text'].mode().head(1))
   

    st.header('Latest Average sentiment count of product')
    plotPieChart(results_df_sorted.head(10))

    st.header('Past Average sentiment count of product')
    plotPieChart(results_df_sorted.tail(10))

