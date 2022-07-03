
import pandas as pd

wk=pd.read_csv('Review_f.csv')
workbook=wk.iloc[:,[3,4]].values
df=pd.DataFrame(workbook)
siz=4000
df1=df[df[0]>=3]
df2=df[df[0]<3]

df1=df1[:siz]
df2=df2[:siz]  

df1=df1[1]
df2=df2[1]

df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)

senti_review =[1]*len(df1)

df1['scores']=senti_review

senti_review=[]
senti_review=[0]*len(df2)

df2['scores']=senti_review
l1=[]

l1=list(df1[1])
l2=list(df2[1])

l=l1+l2



df=pd.DataFrame(l)
seq=[1]*siz+[0]*siz

df['scores']=seq

from sklearn.utils import shuffle
df = shuffle(df)
score=[]
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyse=SentimentIntensityAnalyzer()
for i in df[0]:
    score.append(analyse.polarity_scores(str(i)))
    

senti_review=[]

for i in range(0,len(score)):
    if(score[i]['compound']>=0):
        senti_review.append(1)
    elif(score[i]['compound']<0):
        senti_review.append(0)

count=0
for i in senti_review:
    if i==1:
        count=count+1

from sklearn.metrics import accuracy_score
acc=accuracy_score(df['scores'],senti_review)
print(acc)


score=[]
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyse=SentimentIntensityAnalyzer()
score.append(analyse.polarity_scores(str("The food is good and the atmosphere is nice")))
print(score)
   

