#!/usr/bin/python
# -*-coding: utf-8 -*-
##from __future__ import absolute_import
######
import botnoi as bn
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np


def trainmodel(modelFileName='sentiment.mod'):
    # get data
    goodlist = ['น่ารักมาก', 'สวยจัง', 'ชอบนะ', 'ดีจังเลยนะ', 'สุดยอดไปเลย']
    badlist = ['เฮงซวย', 'ห่วย', 'แย่สุด ๆ ', 'โถ่ ไม่ไหวอ่ะ', 'เชี่ย เอ้ย']
    namestep1 = ['ผมชื่ออรรถวุฒิ', 'ชื่อฟิว',
                 'ชื่อเอฟ', 'ชื่อศิรสิทธิ์', 'ฉันชื่อ']
    agestep2 = ['ผมอายุ 21 ครับ', 'อายุ 30 ค่ะ', 'ฉันอายุ 50 ', '25', '30 ค่ะ']
    tempstep3 = ['36 องศาครับ', '36 องศาค่ะ',
                 '37 ครับ', '37 ค่ะ', '37.5 องศาค่ะครับ']

    # extract feature
    goodfeat = [bn.nlp.text(sen).getw2v_light() for sen in goodlist]
    badfeat = [bn.nlp.text(sen).getw2v_light() for sen in badlist]
    namestep1 = [bn.nlp.text(sen).getw2v_light() for sen in namestep1]
    agestep2 = [bn.nlp.text(sen).getw2v_light() for sen in agestep2]
    tempstep3 = [bn.nlp.text(sen).getw2v_light() for sen in tempstep3]
    # create training set
    nlpdataset = pd.DataFrame()
    nlpdataset['feature'] = goodfeat + \
        badfeat + namestep1 + agestep2 + tempstep3
    nlpdataset['label'] = ['good']*5 + ['bad']*5 + ['ขอทราบอายุคนไข้ครับ'] * \
        5 + ['ขอทราบอุณหภูมิครับ']*5 + \
            ['เเล้ววันนี้คนไข้มีอาการอะไรบ้างครับ']*5
    # train model
    clf = LinearSVC()
    mod = clf.fit(
        np.vstack(nlpdataset['feature'].values), nlpdataset['label'].values)
    # save model
    pickle.dump(mod, open(modelFileName, 'wb'))
    return 'model created'


# load model
mod = pickle.load(open('sentiment.mod', 'rb'))


def get_sentiment(sen):
    feat = bn.nlp.text(sen).getw2v_light()
    res = mod.predict([feat])[0]
    return {'result': res}
