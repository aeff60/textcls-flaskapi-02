#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
######
import botnoi as bn
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np

def trainmodel(modelFileName='sentiment.mod'):
    ### get data
    goodlist = ['น่ารักมาก','สวยจัง','ชอบนะ','ดีจังเลยนะ','สุดยอดไปเลย']
    badlist = ['เฮงซวย','ห่วย','แย่สุด ๆ ','โถ่ ไม่ไหวอ่ะ','เชี่ย เอ้ย']
    ### extract feature
    goodfeat = [bn.nlp.text(sen).getw2v_light() for sen in goodlist]
    badfeat = [bn.nlp.text(sen).getw2v_light() for sen in badlist]
    ### create training set
    nlpdataset = pd.DataFrame()
    nlpdataset['feature'] = goodfeat + badfeat
    nlpdataset['label'] = ['good']*5 + ['bad']*5
    ### train model
    clf = LinearSVC()
    mod = clf.fit(np.vstack(nlpdataset['feature'].values),nlpdataset['label'].values)
    ### save model
    pickle.dump(mod,open(modelFileName,'wb'))
    return 'model created'

### load model
mod = pickle.load(open('sentiment.mod','rb'))
def get_sentiment(sen):
  feat = bn.nlp.text(sen).getw2v_light()
  res = mod.predict([feat])[0]
  return {'result':res}