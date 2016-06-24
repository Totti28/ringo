# coding:utf-8
from janome.tokenizer import Tokenizer
from collections import Counter
import itertools
import sklearn
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import re   #This is for Regular Expression
transformer = TfidfTransformer()
c_all = []
bow_all = []
print 'start'
#filter number

import pickle

#bow_final = 個別篇數對應丟進來的群組文章整體的tf-idf值
"""
#輸出年度平均

"""

f = open('store.pckl')
bow_final = pickle.load(f)
f.close()

e = open('store2.pckl')
words_all = pickle.load(e)
e.close()



from operator import itemgetter

#輸出2011～16的結果＋總平均結果

#總平均
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(len(bow_final)):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_total.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(len(bow_final)))+"\n")
f.close()

#2011平均
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(0,6728):
        bow_year[i]=bow_year[i]+bow_final[a][i]

bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)

sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_2011.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(6728))+"\n")
f.close()

#2012
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(6729,13146):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)

sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_2012.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(6418))+"\n")
f.close()

#2013
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(13147,18321):
        bow_year[i]=bow_year[i]+bow_final[a][i]

bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)

sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_2013.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(5175))+"\n")
f.close()

#2014
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(18322,22600):
        bow_year[i]=bow_year[i]+bow_final[a][i]

bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)

sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_2014.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(4279))+"\n")
f.close()

#2015
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(22601,26388):
        bow_year[i]=bow_year[i]+bow_final[a][i]

bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)

sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_2015.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(3788))+"\n")
f.close()

#2016
bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(26389,30269):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("result_2016.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(3881))+"\n")
f.close()