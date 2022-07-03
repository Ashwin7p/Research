# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xlrd
import pandas as pd
workbook=xlrd.open_workbook('HOME.xlsx')
df=pd.DataFrame()

worksheet=workbook.sheet_by_name('home')
li=[]
k=[]
for i in range(1,1782) :
    k=[worksheet.cell(i,1).value,worksheet.cell(i,2).value]
    li.append(k)
    #d =pd.dataframe(df)
    
df=pd.DataFrame(li)
score=[]
kd=str(df[0])
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyse=SentimentIntensityAnalyzer()
for i in range(0,1780):
    score.append(analyse.polarity_scores(kd))


import nltk
tok=list(df[0])
#toks=str(tok)
#toks=toks.split(' ')
for i in range(0,1780):
    post=nltk.pos_tag(tok)


