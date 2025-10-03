
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯正则版本：爬取 https://www.ft.com/china 页面新闻标题
说明：初版因环境 charset_normalizer 导致 requests / bs4 初始化问题，采用 urllib + 正则。

增强特性：
1. 支持 --proxy 指定 HTTP/HTTPS 代理（例：--proxy http://127.0.0.1:7890）
2. 支持 --retries 与 --retry-delay 控制重试
3. 支持网络失败时 --offline-demo 返回示例标题
4. 可用 --save 保存结果为 JSON 文件（默认不保存）
5. 统一函数 fetch_ft_china_titles 返回标题 list

使用示例：
python 1_FT_china.py --proxy http://127.0.0.1:7890 --retries 3 --save
python 1_FT_china.py --offline-demo
"""
import time
import random
import re
from typing import List
import sys
import json
import argparse
import os
import urllib.request
import urllib.error
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header

FT_CHINA_URL = "https://www.ft.com/china"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

TITLE_KEYWORDS = ["china", "中国", "金融", "市场", "经济", "央行", "美元", "投资", "债", "股"]
NAV_FILTER = ["订阅", "登录", "Register", "Sign in", "Markets data", "Opinion", "数据", "广告"]

A_TAG_PATTERN = re.compile(r'<a\b[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
TAG_STRIP_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')


def send_email(titles: List[str]) -> bool:
    print("send start")
    from_addr = '331854672@qq.com'
    password = os.getenv('QQ_SMTP_TOKEN', 'olecukyijdtsbhij')  # 建议改成环境变量
    to_addr = 'hekai2023@outlook.com'
    smtp_server = 'smtp.qq.com'

    if not titles:
        print("邮件未发送：标题列表为空")
        return False

    # 将列表转换为纯文本正文
    body_lines = [f"{i+1:02d}. {t}" for i, t in enumerate(titles)]
    email_body = "====== FT China 标题列表 ======\n" + "\n".join(body_lines)

    try:
        msg = MIMEText(email_body, 'plain', 'utf-8')
        current_time = time.strftime('%Y-%m-%d', time.localtime())
        msg['From'] = Header(from_addr)
        msg['To'] = Header(to_addr)
        msg['Subject'] = Header(f"ft_{current_time}_news")

        server = smtplib.SMTP_SSL(smtp_server, 465, timeout=15)
        # 调试输出(需要时启用)
        # server.set_debuglevel(1)
        server.login(from_addr, password)
        resp = server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()

        if resp:  # sendmail 返回非空 dict 代表某些地址失败
            print(f"部分收件人失败: {resp}")
            return False

        print("send success")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e.__class__.__name__}: {e}")
        return False

def build_opener(proxy: str = None):
    """构建带或不带代理的 opener"""
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy
        }))
    opener = urllib.request.build_opener(*handlers)
    return opener


def fetch_page(url: str, timeout: int = 15, proxy: str = None, retries: int = 1, retry_delay: float = 1.5) -> str:
    """获取网页 HTML 文本，失败返回空字符串。
    支持重试与代理。
    """
    opener = build_opener(proxy)
    attempt = 0
    while attempt <= retries:
        try:
            time.sleep(random.uniform(0.4, 0.9))
            req = urllib.request.Request(url, headers=HEADERS, method='GET')
            with opener.open(req, timeout=timeout) as r:  # type: ignore
                if r.status == 200:
                    raw = r.read()
                    return raw.decode('utf-8', errors='ignore')
                print(f"[WARN] HTTP {r.status} -> {url}")
                return ""
        except Exception as e:
            print(f"[ERROR] 第 {attempt+1} 次请求失败: {e}")
            attempt += 1
            if attempt <= retries:
                time.sleep(retry_delay * attempt)
    return ""


def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = TAG_STRIP_PATTERN.sub(' ', txt)  # 去除内嵌标签
    txt = txt.replace('&nbsp;', ' ').replace('&amp;', '&')
    txt = WHITESPACE_PATTERN.sub(' ', txt).strip('|· ').strip()
    return txt


def looks_like_title(text: str) -> bool:
    if len(text) < 8:
        return False
    lower = text.lower()
    if any(k.lower() in lower for k in TITLE_KEYWORDS):
        return True
    if re.search(r'[\u4e00-\u9fff]', text) and len(text) >= 10:
        return True
    return False


def offline_demo_titles() -> List[str]:
    now = datetime.now().strftime('%Y-%m-%d')
    return [
        f"中国市场在全球不确定性中表现分化 - {now}",
        f"人民币汇率波动加大 投资者关注央行政策信号 - {now}",
        f"大型科技股再遭审视 监管趋势成焦点 - {now}",
        f"全球资金流向变化 资产配置策略重新评估 - {now}",
        f"大宗商品价格走软 企业成本压力缓解 - {now}",
    ]


def parse_titles(html: str) -> List[str]:
    titles = []
    seen = set()
    for match in A_TAG_PATTERN.finditer(html):
        inner = match.group(1)
        text = clean_text(inner)
        if not looks_like_title(text):
            continue
        if any(f.lower() in text.lower() for f in NAV_FILTER):
            continue
        if text in seen:
            continue
        seen.add(text)
        titles.append(text)

    def score(t: str):
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', t))
        return (chinese_chars * 2) + len(t)

    titles.sort(key=score, reverse=True)
    return titles[:60]


def fetch_ft_china_titles(proxy: str = None, retries: int = 1, retry_delay: float = 1.5, offline_demo: bool = False) -> List[str]:
    if offline_demo:
        return offline_demo_titles()
    html = fetch_page(FT_CHINA_URL, proxy=proxy, retries=retries, retry_delay=retry_delay)
    if not html:
        return []
    return parse_titles(html)


def save_json(titles: List[str], filename: str = None) -> str:
    if not filename:
        filename = f"ft_china_titles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = {
        "source": FT_CHINA_URL,
        "count": len(titles),
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "titles": titles,
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="爬取 FT China 标题")
    p.add_argument('--proxy', help='HTTP/HTTPS 代理，如 http://127.0.0.1:7890')
    p.add_argument('--retries', type=int, default=1, help='请求失败重试次数')
    p.add_argument('--retry-delay', type=float, default=1.5, help='初始重试延迟秒')
    p.add_argument('--offline-demo', action='store_true', help='使用离线演示数据')
    p.add_argument('--save', action='store_true', help='保存结果为 JSON 文件')
    p.add_argument('--output', help='保存文件名（需与 --save 一起使用）')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    proxy = args.proxy or os.environ.get('FT_PROXY')
    if proxy:
        print(f"[INFO] 使用代理: {proxy}")

    titles = fetch_ft_china_titles(proxy=proxy, retries=args.retries, retry_delay=args.retry_delay, offline_demo=args.offline_demo)

    print("====== FT China 标题列表 ======")
    if not titles:
        print("未抓取到标题（可能需要代理或网络受限，或可加 --offline-demo 测试）")
        return titles

    for i, t in enumerate(titles, 1):
        print(f"{i:02d}. {t}")

    if args.save:
        fname = save_json(titles, args.output)
        print(f"\n[INFO] 已保存: {fname}")

    titles = titles[0:10]
    print("send email ")
    send_email(titles)
    return titles


if __name__ == '__main__':
    #main()
    while True:
        try:
            print("start:........")
            main()
            time.sleep( 12 * 60 * 60 )
        except Exception as e:
            print(" ")
            print("cause error:........")
            print("try again after one minute")
            time.sleep(60)
            continue


