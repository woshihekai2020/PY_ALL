# https://blog.csdn.net/2201_76125261/article/details/149200494
import aiohttp
import asyncio
from pyquery import PyQuery as pq
import pandas as pd
from urllib.parse import urljoin
import time


async def fetch_page(session, url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            return await response.text()
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return None


async def parse_news(html):
    news_list = []
    if not html:
        return news_list

    doc = pq(html)

    # 解析头条新闻
    top_items = doc('.top-news-wrap .news-item')
    for item in top_items.items():
        title = item.text()
        link = item.attr('href') or ''
        if link and not link.startswith(('http://', 'https://')):
            link = urljoin('https://news.sina.com.cn', link)
        news_list.append({'title': title, 'link': link, 'type': 'top'})

    # 解析普通新闻
    normal_items = doc('.news-item')
    for item in normal_items.items():
        title = item.text()
        link = item.attr('href') or ''
        if link and not link.startswith(('http://', 'https://')):
            link = urljoin('https://news.sina.com.cn', link)
        news_list.append({'title': title, 'link': link, 'type': 'normal'})

    return news_list


async def crawl_sina_news_async():
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        url = 'https://news.sina.com.cn/'
        html = await fetch_page(session, url)
        news_list = await parse_news(html)

        # 保存数据
        if news_list:
            df = pd.DataFrame(news_list)
            df.to_csv('sina_news_async.csv', index=False, encoding='utf-8-sig')
            print(f"异步爬取完成，共获取{len(df)}条新闻")


async def main():
    await crawl_sina_news_async()


if __name__ == '__main__':
    asyncio.run(main())
