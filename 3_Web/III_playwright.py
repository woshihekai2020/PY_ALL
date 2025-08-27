# https://blog.csdn.net/2201_76125261/article/details/149200494
from playwright.sync_api import sync_playwright
import pandas as pd
import time


def crawl_sina_news_with_playwright():
    news_list = []

    with sync_playwright() as p:
        # 启动浏览器（可选择chromium、firefox或webkit）
        browser = p.chromium.launch(
            headless=False,  # 设置为True则不显示浏览器窗口
            slow_mo=100,  # 减慢操作速度，便于观察
        )

        # 创建新页面
        page = browser.new_page(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )

        try:
            # 导航到目标页面
            page.goto('https://news.sina.com.cn/', timeout=60000)

            # 等待关键元素加载
            page.wait_for_selector('.top-news-wrap', timeout=10000)

            # 模拟滚动以加载更多内容
            for _ in range(3):
                page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                time.sleep(1)

            # 提取头条新闻
            top_news = page.query_selector_all('.top-news-wrap .news-item')
            for item in top_news:
                title = item.inner_text()
                link = item.get_attribute('href')
                news_list.append({'title': title, 'link': link, 'type': 'top'})

            # 提取普通新闻
            normal_news = page.query_selector_all('.news-item')
            for item in normal_news:
                title = item.inner_text()
                link = item.get_attribute('href')
                news_list.append({'title': title, 'link': link, 'type': 'normal'})

            # 保存数据
            df = pd.DataFrame(news_list)
            df.drop_duplicates(subset=['link'], inplace=True)
            df.to_csv('sina_news_playwright.csv', index=False, encoding='utf-8-sig')
            print(f"使用Playwright成功爬取{len(df)}条新闻")

        except Exception as e:
            print(f"爬取过程中出错: {str(e)}")
        finally:
            browser.close()


if __name__ == '__main__':
    crawl_sina_news_with_playwright()
