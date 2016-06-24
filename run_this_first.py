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
words_all = []
c_all = []
bow_all = []
print 'start'
#filter number
delimiters = u'０１２３４５６７８９'
word_attribute = [u'名詞', u'動詞', u'形容詞', u'カスタム名詞']
#filter word
specialWords = [u'福島', u'する', u'いる', u'できる', u'なる', u'れる', u'ある', u'せる', u'くる', u'よる', u'みる', u'いく',
                u'*', u'人', u'県', u'市', u'日', u'月', u'さん', u'氏', u'社', u'ため', u'こ', u'そ', u'地', u'住', u'学', u'度',
                u'の', u'こと', u'％', u'町', u'長', u'部', u'校', u'省', u'性', u'化', u'民', u'高', u'大', u'小', u'底',
                u'村', u'者', u'たち', u'よう', u'ない', u'的', u'いう', u'員', u'面', u'台', u'午', u'区', u'本',
                u'中', u'後', u'前', u'東', u'南', u'西', u'北', u'上', u'下', u'先', u'今', u'内', u'ら',
                u'分',  u'時', u'年', u'歳', u'万', u'億', u'円', u'千', u'所', u'会', u'回', u'ない', u'さ', u'百', u'回', u'駅', u'件',
                u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十']

t = Tokenizer("user_simpledic.csv", udic_type="simpledic", udic_enc="utf8")
f = open('All_year_text.txt', 'r')
lines = f.readlines()
print 'readline'
for line in lines:
    # Replace certain words with empty space
    for word in specialWords:
        if word in line.decode('utf-8'):
            line = (line.decode('utf-8').replace(word,u' ')).encode('utf-8')
    #line = (line.decode('utf-8').replace(u'福島',u' ')).encode('utf-8')
    # Replace number from string
    for d in delimiters:
        if d in line.decode('utf-8'):
            line = (line.decode('utf-8').replace(d,u' ')).encode('utf-8')

    bow_line = []


    for token in t.tokenize(line.decode('utf-8')):
       if  (token.part_of_speech.split(",")[0] in word_attribute):
            if token.base_form not in specialWords:
                bow_line.append(token.base_form)


# print len(bow_line)
    C = Counter(bow_line)

    c_line = {}
    for k,v, in C.items():

        c_line[k] = v
        words_all.append(k)
    c_all.append(c_line)
        #print c_line

        #print k,c_line[k]
#print c_all
#print len(new_c_all)
#print new_c_all


words_all =list(set(words_all))

import pickle



for dic in c_all:
    bow_line = [0] * len(words_all)
    #print len(words_all)
    for k,v in dic.items():
        n = words_all.index(k)
        #print k,n
        bow_line[n] = v

    bow_all.append(bow_line)
# print len(bow_all)
'''
import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x=v1[i]
        y=v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

'''
print 'start tfidf'
tfidf = TfidfTransformer()
tfidf.fit(bow_all)

tf_idf_matrix = tfidf.transform(bow_all)
bow_new = tf_idf_matrix.todense()
bow_final=bow_new.tolist()

import pickle

f = open('store.pckl', 'w')
pickle.dump(bow_final, f)
f.close()

e = open('store2.pckl', 'w')
pickle.dump(words_all, e)
e.close()
#bow_final = 個別篇數對應丟進來的群組文章整體的tf-idf值
"""
#輸出年度平均

"""

print 'end tfidf'

#bow_year = 將bow_final裡面的全部文章的tf-idf值作加總後的單一listf2.close()'''