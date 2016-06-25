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
'''
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
'''
#2011個別報社

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(0,1342):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("朝日11.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(1342))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(1343,2649):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("讀賣11.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(1306))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(2650,3772):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("每日11.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(1122))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(3773,4830):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("河北11.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(1058))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(4831,6728):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("福島民報11.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(1898))+"\n")
f.close()

#2012各報社

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(6729,7540):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("朝日12.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(812))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(7541,8503):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("每日12.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(963))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(8504,9238):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("讀賣12.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(735))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(9239,9523):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("河北12.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(285))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(9524,13145):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("福島民報12.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(3622))+"\n")
f.close()

#2013各報社

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(13146,13701):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("朝日13.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(556))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(13702,14328):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("每日13.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(627))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(14329,14848):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("讀賣13.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(520))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(14849,15626):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("河北13.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(778))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(15627,18320):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("福島民報13.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(2694))+"\n")
f.close()

#2014 各報社


bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(18321,18764):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("朝日14.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(444))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(18765,19191):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("每日14.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(427))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(19192,19544):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("讀賣14.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(353))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(19545,20233):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("河北14.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(689))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(20234,22600):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("福島民報14.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(2367))+"\n")
f.close()

#2015 各報社


bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(22601,23002):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("朝日15.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(402))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(23003,23404):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("每日15.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(402))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(23405,23695):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("讀賣15.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(291))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(23696,24265):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("河北15.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(570))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(24266,26388):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("福島民報15.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(2123))+"\n")
f.close()

#2016 各報社

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(26389,26837):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("朝日16.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(449))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(26838,27259):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("每日16.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(422))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(27260,27573):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("讀賣16.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(314))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(29449,30269):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("河北16.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(821))+"\n")
f.close()

bow_year=[]
for a in range(len(words_all)):
    bow_year.append(0.0)
for i in range(len(words_all)):
    for a in range(27574,29448):
        bow_year[i]=bow_year[i]+bow_final[a][i]
bow_2=[]
for i in range(len(words_all)):
    a=[]
    a.append(words_all[i])
    a.append(bow_year[i])
    bow_2.append(a)
sorted_final = sorted(bow_2, key=itemgetter(1), reverse = True)
#sorted_final = bow_year內的值排序後之結果
f=open("福島民報16.txt","w")
for i in range(len(sorted_final)):
    f.write(sorted_final[i][0].encode('utf-8')+" : "+str(sorted_final[i][1]/float(1875))+"\n")
f.close()
