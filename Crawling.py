#-*- encoding:utf-8 -*-
import requests
import json
import re
from bs4 import BeautifulSoup

seed = "https://cloud.tencent.com/document/product/1709"
baseUrl="https://cloud.tencent.com"
appendUrlList=[]
appendDataList = []

# 获取各栏目URL
def getCrawl(seed):
    seeds = []
    seeds.append(seed)
    textdata = requests.get(seed).text
    soup = BeautifulSoup(textdata,'lxml')
    nodes = soup.select("textarea.J-qcSideNavListData")
    jsonObj=json.loads(nodes[0].getText())["list"]
    seeds.append(nodes)
    getChild(jsonObj) 

def getChild(nowObj):
     if nowObj is not None:
        for n in nowObj:
            links= baseUrl+n["link"]
            data={"title":n["title"],"link":links}
            appendUrlList.append(data)
            if n.get("children") is not None:
                getChild(n.get("children"))
                
                
def crawlData():
    getCrawl(seed)
    count=0
    for data in appendUrlList:
        url = data["link"]
        print(data["title"]+"        "+data["link"])
        # print(url)
        textdata = requests.get(url).text
        soup = BeautifulSoup(textdata,'lxml')
        nodes = soup.select("div.J-markdown-box")
        if nodes is not None and len(nodes)>0:
            text = nodes[0].get_text()
            text = text[:6000]
            stringText = re.sub('\n+', '\n', text)
            data={"url":url,"title":data["title"],"text":stringText}
            appendDataList.append(data)
        # count=count+1
        # if count>6:
    return appendDataList


# print(len(appendDataList))