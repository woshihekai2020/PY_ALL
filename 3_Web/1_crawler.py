#encoding=utf-8
import requests
from bs4 import BeautifulSoup
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import datetime
import os
from urllib3.util import current_time


class newNotice( object ):
    def __init__(self):
        self.url = 'https://www.ft.com/china'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        self.notice = ''

    def get_response(self, url):
        print(url)
        print("before request")
        response = requests.get(url)
        print("before data")
        data = response.content
        print(data)
        return data


    def data_save(self, data):
        with open('C.html', 'wb') as f:
            f.write(data)

    def parse_data(self, data):
        soup = BeautifulSoup(data, 'html.parser', from_encoding='gb18030')
        all  = soup.find( id = 'stream' )
        new_url = self.url
        return new_url

    def get_time(self, RAWcur):
        dict ={'January':1, 'Feburary':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
        a = RAWcur.find( ',' )
        b = RAWcur.find( ' ' )
        tmp = RAWcur.split( ',', 2 )
        tmpM = tmp[1].split( ',', 2)
        month = tmpM[2]
        curTime = tmpM[1] + " " + str(dict[month]) + ", " + str(tmp[2])
        cur_need_time = datetime.datetime.strptime(curTime, '%d %m, %Y').strftime('%Y-%m-%d')
        return cur_need_time

    def get_content(self, new_url):
        data = requests.get(new_url).content
        soup = BeautifulSoup(data, 'html.parser', from_encoding='gb18030')
        content = soup.find('div', class_='o-teaser__heading').text[0: -1]
        print("content:", content)
        content_all = soup.find_all( 'div', class_ = 'o-teaser__heading' )
        for CONTETN  in content_all:
            self.notice += '\n' + CONTETN.text + '\n'
        print( self.notice )

        new_time_str = soup.find('div', class_="stream-card__date").text[0: -1]
        news_time = self.get_time(new_time_str)

        current_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        print( current_time )

        if news_time.strip() == current_time.strip():
            print("今日有新公告")
            self.notice = news_time + "有新公告:" + self.notice
            self.send_email(self.notice)
        else:
            print("今日无新公告")
            self.notice = "今日无新公告"
            #self.send_email(self.notice)
    def send_email(self, content):
        from_addr = '331854672@qq.com'
        password = 'peuoyttgaiyvbieb'

        to_addr = 'hekai2023@outlook.com'

        smtp_server = 'smtp.qq.com'

        msg = MIMEText(content, 'plain', 'utf-8')

        msg['From'] = Header(from_addr)
        msg['To'] = Header(to_addr)
        msg['Subject'] = Header('New Notice')

        server = smtplib.SMTP_SSL(smtp_server)
        server.connect(smtp_server, 465)

        server.login(from_addr, password)

        server.sendmail(from_addr, to_addr, msg.as_string())

        server.quit()

    def run(self):
        try:
            print( "1" )
            data = self.get_response(self.url)
            print("2")
            new_url = self.parse_data(data)
            print( "new_url: ", new_url)
            self.get_content(new_url)
        except:
            print("error: reRUN")
            return 1
if __name__ == "__main__":
    #while True:
        #newNotice().run()

        #time.sleep(43200)
    # 目标 URL

    url = 'https://www.ft.com/china'
    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    # 发送请求
    response = requests.get(url, headers=headers)
    # 检查请求是否成功
    if response.status_code == 200:
        # 解析 HTML 内容
        soup = BeautifulSoup(response.content, 'html.parser')
        # 查找新闻标题
        articles = soup.find_all('div', class_='o-teaser__heading')
        # 打印新闻标题
        for article in articles:
            title = article.text.strip()
            print(title)
    else:
        print(f"请求失败，状态码: {response.status_code}")





#https://blog.csdn.net/qq_37457202/article/details/106537627
#定时爬取
#https://blog.csdn.net/weixin_28475533/article/details/113495447
