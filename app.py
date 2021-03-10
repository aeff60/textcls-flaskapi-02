#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
######
import json
import time
from flask import Flask
from flask_restful import Resource, Api, reqparse
import requests
import predict_sentiment as ps
import pandas as pd

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return "Hello World!"

class get_sentiment(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('keyword', type=str)
        dictp = parser.parse_args()
        kw = dictp['keyword']
        res = ps.get_sentiment(kw)
        return res

api.add_resource(get_sentiment, '/get_sentiment',endpoint='get_sentiment')
if __name__ == '__main__':
    app.run(threaded=True)
