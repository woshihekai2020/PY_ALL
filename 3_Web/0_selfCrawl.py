import requests
from bs4 import BeautifulSoup
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import datetime


class newNotice(object):
    def __init__(self):
        # self.url = 'https://www.ft.com/world/asia-pacific/china'
        self.url = 'https://www.ft.com/china'
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"
        }
        self.notice = ''

    def get_response(self, url):
        print(url)
        response = requests.get(url)
        data = response.content
        return data

    def data_save(self, data):
        with open('C.html', 'wb') as file:
            file.write(data)

    def parse_data(self, data):
        soup = BeautifulSoup(data, 'html.parser', from_encoding='gb18030')
        all = soup.find(id="stream")
        new_url = self.url  # +all.a['href']
        # print( self.url )
        # print(all.a.text)
        # self.notice = all.a.text
        # print(new_url)
        return new_url

    def get_time(self, RAWcur):
        dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        # RAWcur = 'Wednesday, 12 January, 2022'
        a = RAWcur.find(',')
        b = RAWcur.find(' ')
        tmp = RAWcur.split(',', 2)  # exp:['Wednesday', ' 12 January', ' 2022']
        tmpM = tmp[1].split(' ', 2)  # exp:'', '12', 'January']
        month = tmpM[2]  # exp:January
        # print( dict[month] )     #int : 1
        curTime = tmpM[1] + " " + str(dict[month]) + ", " + str(tmp[2])
        cur_need_time = datetime.datetime.strptime(curTime, '%d %m,  %Y').strftime('%Y-%m-%d')
        # print( cur_need_time ) #exp:2022-01-12
        return cur_need_time

    def get_content(self, new_url):
        data = requests.get(new_url).content
        soup = BeautifulSoup(data, 'html.parser', from_encoding='gb18030')
        content = soup.find('div', class_="o-teaser__heading").text[0:-1]
        # content_all = soup.find('div',class_="o-teaser__heading").text[0:-1]
        content_all = soup.find_all('div', class_='o-teaser__heading')
        # print( content_all )
        # print( content_all[00].string )
        for CONTENT in content_all:
            # print( CONTENT.string )
            print(CONTENT.text)
            self.notice = self.notice + CONTENT.text + '\n' + '\n'

        # self.notice = content
        # print(content)

        news_time_str = soup.find('div', class_="stream-card__date").text[0:-1]  #
        # print( news_time_str  )
        news_time = self.get_time(news_time_str)
        # print( news_time )

        current_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        print(current_time)

        if news_time.strip() == current_time.strip():
            print("今日有新公告: " + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
            # print(self.notice)
            # self.notice = "今日_" + current_time +"_有新公告:" + '\n' + self.notice
            # self.notice = self.notice
            self.send_email(self.notice)
            time.sleep(12 * 60 * 60)
        else:
            print("今日没有新公告: " + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
            self.notice = "今日没有新公告"
            # self.send_email(self.notice)
            time.sleep(60 * 60)

    def send_email(self, email_body):
        from_addr = '331854672@qq.com'
        password = 'uhlstkbcftknbidj'  # 这里是QQ邮箱授权码

        # 收信方邮箱
        to_addr = 'hekai2023@outlook.com'

        # 发信服务器
        smtp_server = 'smtp.qq.com'

        # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
        msg = MIMEText(email_body, 'plain', 'utf-8')

        current_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))

        # 邮件头信息
        msg['From'] = Header(from_addr)
        msg['To'] = Header(to_addr)
        msg['Subject'] = Header('ft_' + current_time + '_news')

        # 开启发信服务，这里使用的是加密传输
        server = smtplib.SMTP_SSL(smtp_server)
        server.connect(smtp_server, 465)
        # 登录发信邮箱
        server.login(from_addr, password)
        # 发送邮\E4\BB
        server.sendmail(from_addr, to_addr, msg.as_string())
        # 关闭服务器
        server.quit()

    def run(self):
        data = self.get_response(self.url)
        new_url = self.parse_data(data)
        self.get_content(new_url)


# newNotice().run()


if __name__ == "__main__":

    #	while True:
    #		print( "start:........" )
    #		newNotice().run()
    #		time.sleep( 12 * 60 * 60 )

    while True:
        try:
            print("start:........")
            newNotice().run()
            # time.sleep( 12 * 60 * 60 )
        except Exception as e:
            print(" ")
            print("cause error:........")
            print("try again after one minute")
            time.sleep(60)
            continue

# https://blog.csdn.net/qq_37457202/article/details/106537627


# 定时爬取
# https://blog.csdn.net/weixin_28475533/article/details/113495447
