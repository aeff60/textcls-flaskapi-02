#!/usr/bin/python
# -*-coding: utf-8 -*-
# from __future__ import absolute_import
######
import botnoi as bn
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
from flask import jsonify


def trainmodel(modelFileName='sentiment.mod'):
    # get data
    namelist = ['ผมชื่ออรรถวุฒิ', 'ชื่อฟิว',
                'ชื่อเอฟ', 'ชื่อศิรสิทธิ์', 'ฉันชื่อ']
    agelist = ['ผมอายุ 21 ครับ', 'อายุ 30 ค่ะ', 'ฉันอายุ 50 ', '25', '30 ค่ะ']
    templist = ['36 องศาครับ', '36 องศาค่ะ',
                '37 ครับ', '37 ค่ะ', '37.5 องศาค่ะครับ']
    headachelist = ['ไม่มีอาการปวดหัว', 'มีอาการปวดหัวเล็กน้อยครับ',
                    'ไม่ปวดหัวนะครับ', 'ไม่ปวดหัว', 'ปวดหัวนิดหน่อย']

    # extract feature
    name = [bn.nlp.text(sen).getw2v_light() for sen in namelist]
    age = [bn.nlp.text(sen).getw2v_light() for sen in agelist]
    temp = [bn.nlp.text(sen).getw2v_light() for sen in templist]
    headache = [bn.nlp.text(sen).getw2v_light() for sen in headachelist]
    # create training set
    nlpdataset = pd.DataFrame()
    nlpdataset['feature'] = name + age + temp + headache
    nlpdataset['label'] = ['ขอทราบอายุคนไข้ครับ'] * 5 + ['ขอทราบอุณหภูมิครับ'] * \
        5 + ['เเล้ววันนี้คนไข้มีอาการอะไรบ้างครับ']*5 + ['0']*5
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
    # return jsonify(res)
