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

    templist1 = ['อุณหภูมิ37.0องศาครับ', 'อุณหภูมิ37.5องศาค่ะ',
                 'อุณหภูมิ37.8องศาค่ะ', 'อุณหภูมิ38องศาครับ', 'อุณหภูมิ38.5องศา']
    templist2 = ['อุณหภูมิ38.9องศาครับ', 'อุณหภูมิ39.0องศาค่ะ',
                 'อุณหภูมิ39.2องศา', 'อุณหภูมิ39.5องศาครับ', 'อุณหภูมิ39.5องศาค่ะ']
    templist3 = ['อุณหภูมิ39.6องศาค่ะ', 'อุณหภูมิ39.7องศา',
                 'อุณหภูมิ39.8องศา', 'อุณหภูมิ40องศา', 'อุณหภูมิ40.3องศา']
    # 0 ไม่เป็น 1 เป็น
    headachelist0 = ['ไม่มีอาการปวดหัว', 'ไม่มีอาการปวดหัวครับ',
                     'ไม่ปวดหัวนะครับ', 'ไม่ปวดหัว', 'ไม่ปวดหัวเลย']
    headachelist1 = ['มีอาการปวดหัว', 'มีอาการปวดหัวเล็กน้อยครับ',
                     'ปวดหัวครับ', 'ปวดเเต่ไม่มาก', 'ปวดหัวนิดหน่อย']
    # extract feature
    temp1 = [bn.nlp.text(sen).getw2v_light() for sen in templist1]
    temp2 = [bn.nlp.text(sen).getw2v_light() for sen in templist2]
    temp3 = [bn.nlp.text(sen).getw2v_light() for sen in templist3]

    headache0 = [bn.nlp.text(sen).getw2v_light() for sen in headachelist0]
    headache1 = [bn.nlp.text(sen).getw2v_light() for sen in headachelist1]
    # create training set
    nlpdataset = pd.DataFrame()
    nlpdataset['feature'] = temp1 + temp2 + temp3 + headache0 + headache1
    nlpdataset['label'] = ['0'] * 5 + ['1'] * \
        5 + ['2'] * 5 + ['0']*5 + ['1']*5
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
