import os

import requests
from dotenv import load_dotenv


load_dotenv()

LAW_OC = os.getenv("LAW_OC")

if not LAW_OC:
    raise ValueError("LAW_OC가 .env에 설정되어 있지 않습니다.")


BASE_URL = "https://www.law.go.kr/DRF"


def search_law(query: str):
    """
    국가법령정보센터에서 법령 이름을 검색합니다.

    예:
        search_law("소득세법")
    """

    url = f"{BASE_URL}/lawSearch.do"

    params = {
        "OC": LAW_OC,
        "target": "law",
        "type": "JSON",
        "query": query,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()
def get_law_detail(mst: str):
    """법령일련번호(MST)를 이용해 법령 본문을 조회합니다."""

    url = f"{BASE_URL}/lawService.do"

    params = {
        "OC": LAW_OC,
        "target": "law",
        "type": "JSON",
        "MST": mst,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()
def parse_articles(law_detail: dict):
    """
    국가법령정보센터 법령 본문 JSON에서
    실제 조문들을 검색 가능한 문서 형태로 변환합니다.
    """

    article_units = law_detail["법령"]["조문"]["조문단위"]

    documents = []

    for article in article_units:
        article_number = article.get("조문번호", "")
        article_text = article.get("조문내용", "")
        article_type = article.get("조문여부", "")

        # 내용이 없는 데이터는 제외
        if not article_text:
            continue

        # 장/절 제목 등은 제외하고 실제 조문만 사용
        if article_type != "조문":
            continue

        documents.append({
            "article_number": article_number,
            "text": article_text.strip()
        })

    return documents