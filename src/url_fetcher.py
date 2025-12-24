import requests
from newspaper import Article
from urllib.parse import urlparse
from bs4 import BeautifulSoup

def fetch_article_text(url, timeout=10):
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "http://" + url
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if text and len(text.strip()) > 50:
            return text.strip()
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        
        paragraphs = soup.find_all("p")
        content = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        if content and len(content) > 50:
            return content
        return None
    except Exception as e:
        return None
