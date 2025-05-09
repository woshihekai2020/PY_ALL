#https://juejin.cn/post/7000554762378674183
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def fetchUrl(url):

    header = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    }

    r = requests.get(url, headers = header)
    r.encoding = r.apparent_encoding
    return r.text

def parseHtml(html):

    bsObj = BeautifulSoup(html, "lxml")
    temp = bsObj.find_all("h2", attrs={"class": "item-headline"})

    retData = []

    for item in temp[0:-4]:

        a = item.find_all("a")[-1]

        if "premium" not in a['href'] and "story" not in a['href'] and "interactive" not in a['href']:
            continue

        link = "https://www.ft.com/" + a["href"]
        title = a.text

        # print(title, link)
        retData.append([title, link])

    return retData

def getDateTime(url):
    # 获得文章的发布日期

    html = fetchUrl(url)
    # print(html)
    bsObj = BeautifulSoup(html, "lxml")
    span = bsObj.find("span", attrs={"class": "story-time"})

    if span:
        pattern = r"(\d+年\d+月\d+日)"
        date = re.findall(pattern, span.text)[0]
        return date
    else:
        if "archive" in url:
            print("no date time")
            return ""
        elif "exclusive" in url:
            url = url.replace("exclusive", "archive")
            return getDateTime(url)
        else:
            url = url + "?exclusive"
            return getDateTime(url)

def saveData(data, filename, mode):

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(filename, mode=mode, index=False, sep=',', header=False, encoding="utf_8_sig")


if __name__ == "__main__":

    # 由于主页上没有发布日期的数据
    # 所以先将 标题 和 链接 保存在 temp.csv 临时文件中
    # 然后再遍历临时文件中的每一条链接，获取其发布日期数据
    print("爬虫启动")
    url = "https://www.ft.com/"
    html = fetchUrl(url)
    data = parseHtml(html)
    saveData(data, "temp.csv", 'w')
    print("临时文件保存成功")

    print("开始爬取详细信息……")
    df = pd.read_csv('temp.csv')
    for index, title, link in df.itertuples():
        print("正在爬取： ", link)
        date = getDateTime(link)
        print(date, title, link)
        saveData([[date, title, link]], "FT新闻.csv", 'a')
        print("----"*20)
        # time.sleep(1)













# import requests

# # Define the URL to crawl
# url = 'https://www.ft.com/'

# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',"Content-Type": "application/json"}

# try:
#     # Send a GET request to the URL
#     response = requests.get(url, headers=headers)

#     # Check if the request was successful
#     if response.status_code == 200:
#         # Print the HTML content of the page
#         print(response.text)
#     else:
#         print('Failed to retrieve the page')
# except requests.exceptions.ReqiuestExcepton as e:
#     print(f'An error occurred: {e}')
