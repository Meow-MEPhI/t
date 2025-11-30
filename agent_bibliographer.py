
import requests
from bs4 import BeautifulSoup
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage


class BibliographerAgent:
    """Агент для извлечения текста научной статьи из URL."""

    def __init__(self, auth_key: str):
        self.auth_key = auth_key
        self.model = GigaChat(credentials=auth_key, verify_ssl_certs=False)

    def fetch_article_text(self, url: str) -> str:
        """Получает и парсит текст статьи по URL."""
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = (soup.find('div', class_='article__body') or soup).get_text(' ', strip=True)
        return text

    def run(self, state: dict) -> dict:

        article_url = state.get("article_url", "")

        article_text = self.fetch_article_text(article_url)

        print(article_text)

        return {
            "article_text": article_text,
            "status": "text_extracted"
        }
