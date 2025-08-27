# https://blog.csdn.net/2201_76125261/article/details/149200494
from requests_html import HTMLSession
import pandas as pd
from urllib.parse import urljoin
import time
import random


def crawl_sina_news_with_requests_html():
    # 初始化会话
    session = HTMLSession()

    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }

    # 目标URL
    url = 'https://news.sina.com.cn/'

    try: 
        # 发送请求
        response = session.get(url, headers=headers, timeout=10)

        # 执行JavaScript渲染（最多等待2秒）
        response.html.render(timeout=20, sleep=2)

        # 解析新闻数据
        news_list = []

        # 解析头条新闻
        top_news = response.html.find('.top-news-wrap .news-item', first=False)
        for item in top_news or []:
            title = item.text
            link = item.absolute_links.pop() if item.absolute_links else ''
            news_list.append({'title': title, 'link': link, 'type': 'top'})

        # 解析普通新闻
        normal_news = response.html.find('.news-item', first=False)
        for item in normal_news or []:
            title = item.text
            link = item.absolute_links.pop() if item.absolute_links else ''
            news_list.append({'title': title, 'link': link, 'type': 'normal'})

        # 转换为DataFrame并保存
        df = pd.DataFrame(news_list)
        df.to_csv('sina_news_requests_html.csv', index=False, encoding='utf-8-sig')
        print(f"成功爬取{len(df)}条新闻数据")

    except Exception as e:
        print(f"爬取失败: {str(e)}")
    finally:
        session.close()


if __name__ == '__main__':
    crawl_sina_news_with_requests_html()
