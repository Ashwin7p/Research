# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import os


os.environ['STANFORD_PARSER'] ='/home/adarsh/Documents/Aspect Extraction/stanford-parser-full-2015-04-20/jars'
os.environ['STANFORD_MODELS'] = '/home/adarsh/Documents/Aspect Extraction/stanford-parser-full-2015-04-20/jars'
os.environ['JAVA_HOME']='/usr/share/Java/jdk1.8.0_161/'


from nltk.parse.stanford import StanfordParser 
parser=StanfordParser(model_path="/home/adarsh/Documents/Aspect Extraction/stanford-parser-full-2015-04-20/jars/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
s="It's really awesome, crystal clear voice and good bass, you can operate it easily. Those user have FM connectivity problem, i would like to say, it's is working for me, just read the manual and try."

sentences=tuple(parser.raw_parse((s)))


for line in sentences:
    for sentence in line:
        sentence.draw()
print(sentences)
import nltk

tree = nltk.tree.Tree.fromstring(str(sentences))


from nltk.parse.stanford import StanfordDependencyParser
dep_parser=StanfordDependencyParser(model_path="/home/adarsh/Documents/Aspect Extraction/stanford-parser-full-2015-04-20/jars/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")



from senticnet.senticnet import SenticNet


aux_verb=['be' ,'am', 'are', 'is', 'was', 'were', 'being', 'been', 'can', 'could', 'dare', 'do' ,'does', 'did', 'have', 'has', 'had', 'having', 'may', 'might', 'must', 'need', 'ought', 'shall', 'should', 'will', 'would']
verb=['VB','VBD','VBG','VBN','VBP','VBZ']
adverb=['RB','RBR','RBS']
noun=['NN','NNS','NNP','NNPS']
adjective=['JJ','JJR','JJS']
dep_rel=''
depe_rel=''
xc=''
z=[]

sample1=[]
import xlrd
workbook=xlrd.open_workbook('reviews.xlsx')
worksheet=workbook.sheet_by_name('Sheet1')
li=[]
k=[]
for i in range(1,191) :
    k=[worksheet.cell(i,1).value]
    sample1.append(k)
    
import nltk
import re
sample=[]
#sample1=['The phone is good. When we hold the phone it feels premium, touch sensitivity is good. Camera quality is pretty good. The battery backup comes for a day with the normal usage','The touch has very smooth functionality. For my business calls and internet usage, I am able to get 2 days back up easily. The charging time is also much better than my previous vivo phone','Quality is not up to the mark. The charging time consuming is more and while using it this drains fast. Heating problem. The packaging is worst and the data cable came with this product was already damaged and without inspecting they packed and dispatched.. I recommend not to buy this','Very big to hold']

for size in range(len(sample1)):
    sample.append(" ".join(re.findall("[a-zA-Z]+", (str(sample1[size]).lower()))))


def sem(d):
    try:
       sn=SenticNet()
       sn.semantics(d)
       return True
    except KeyError :
       return False


k=[]
s=[]
bool1=[]
for lm in range(0,len(sample)):
    bool1.append(True)
from nltk.tokenize import word_tokenize
for gh in range(len(sample)):
   s=sample[gh]
   word_list=word_tokenize(s)
   for c in range(len(aux_verb)):
       if aux_verb[c] in word_list:
          bool1[gh]=False
          break
          



for i in range(0,len(sample)):
    result=dep_parser.raw_parse(sample[i])
    dep=result.__next__()
    k.append(list(dep.triples()))

xc=''   
size=0
iac=[]
z=[]

for loop in range(0,len(k)):
    for i in k[loop]:
        if (i[1]=='amod'):
            z.append(i[0][0])

        if bool1==True:
            if i[1]=='nsubj':
                if i[2][1] in noun :
                    z.append(i[2][0])
            if (i[1]=='xcomp'or i[1]=='advmod') and i[0][1] in verb and (i[2][1] in adjective or i[2][1] in adverb):
                z.append(i[0][0])
                
            if(i[1]=='dobj'and i[0][1] in verb and i[2][1] in noun and (not sem(i[2][0]))):
                dep_rel=i[2][0]
                z.append(i[2][0])
            if(i[1]=='dobj' and i[0][1] in verb and i[2][1] in noun and sem(i[2][0])):
                dep_rel=i[2][0]
                z.append(i[2][0])
                if(i[0][1] in noun and i[2][1] in noun):
                   if dep_rel == i[0][0]:
                     z.append(i[2][0])
                   if dep_rel==i[2][0]:
                     z.append(i[0][0])
        if(i[1]=='xcomp' and i[0][1] in verb):
            
            xc=i[2][0]
        if i[0][0]==xc and i[2][1] in noun and i[1]!='nmod':
            z.append(i[2][0])
        if i[1]=='cop' and i[2][1] in verb:
            z.append(i[0][0])
            for j in k[loop]:
                if j[1]=='nsubj' and j[2][1] in noun:
                    z.append(j[2][0])#in mapping may be there we cant find nice 
    
            depe_rel=i[0][0]
        if(i[0][0]==depe_rel and i[2][1] in verb ):
             z.append(depe_rel)
             z.append(i[2][0])
         
        if(i[1]=='xcomp' and (i[0][1] in adjective or i[0][1] in adverb)):
             z.append(i[0][0])
             
        if i[1]=='nmod' and i[2][1] in noun :
             z.append(i[0][0])
             
        if(i[1]=='dobj'):
             z.append(i[2][0])
    
   
    q=set(z)
    z=list(q)
    iac.append(z)
    z=[]
    

for loop1 in range(0,len(k)):
   z=[] 
   for ad in k[loop1]:
      if(ad[1]=='compound' and ad[0][1] in noun and ad[2][1] in noun):
          z.append(ad[0][0])
          z.append(ad[2][0]) 
          
      if(ad[1]=='conj' and (ad[0][0] in iac[loop1] or ad[0][0] in z)):
          z.append(ad[2][0])
                    
   if(len(z)!=0):  
        q=set(z)
          #print(q,"for 1")
        z=list(q)
        iac[loop1].extend(z)
        z=[]
         
        
   se=set(iac[loop1])
   iac[loop1]=[]
   iac[loop1].extend(list(se))
  
      
          
import lex



aspect=[]
asp=[]
for i in range(len(iac)):
    for j in iac[i]:
        for ke in lex.dict1:
            key=ke
            if j in lex.dict1[key]:
                asp.append(key)
        post=nltk.pos_tag(j.split())
        if post[0][1] in noun:
             if(len(j)>2):
               asp.append(j)
    q=set(asp)
    asp=list(q)            
    aspect.append(asp)
    asp=[]
                

