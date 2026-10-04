import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

from dotenv import load_dotenv
from langchain_core.documents import Document


# --------------------------------------------------
# 환경변수
# --------------------------------------------------

load_dotenv()

LAW_API_OC = os.getenv("LAW_API_OC")


# --------------------------------------------------
# 법령 데이터 가져오기
# --------------------------------------------------

def fetch_law_data(target_law_name="소득세법"):

    if not LAW_API_OC:
        raise ValueError(
            ".env 파일에서 LAW_API_OC를 찾을 수 없습니다."
        )

    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    # --------------------------------------------------
    # 1. 법령 검색
    # --------------------------------------------------

    encoded_query = quote(target_law_name)

    search_url = (
        "https://www.law.go.kr/DRF/lawSearch.do"
        f"?OC={LAW_API_OC}"
        f"&target=law"
        f"&type=XML"
        f"&query={encoded_query}"
    )

    print("\n[법령 검색 API 요청]")
    print(f"법령명: {target_law_name}")

    response = requests.get(
        search_url,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    try:
        root = ET.fromstring(response.content)

    except ET.ParseError as e:

        print(f"XML 파싱 에러: {e}")
        return []


    # --------------------------------------------------
    # 2. 법령 ID(MST) 찾기
    # --------------------------------------------------

    law_id = None

    for elem in root.iter():

        if elem.tag in [
            "법령일련번호",
            "MST",
            "lawId",
            "target_id"
        ]:

            if elem.text:

                law_id = elem.text.strip()
                break


    if not law_id:

        print(
            f"'{target_law_name}' 검색결과에서 "
            "법령 ID를 찾지 못했습니다."
        )

        return []


    print(f"발견된 법령 ID: {law_id}")


    # --------------------------------------------------
    # 3. 법령 상세 데이터 요청
    # --------------------------------------------------

    detail_url = (
        "https://www.law.go.kr/DRF/lawService.do"
        f"?OC={LAW_API_OC}"
        f"&target=law"
        f"&MST={law_id}"
        f"&type=XML"
    )

    detail_resp = requests.get(
        detail_url,
        headers=headers,
        timeout=10
    )

    detail_resp.raise_for_status()


    try:

        detail_root = ET.fromstring(
            detail_resp.content
        )

    except ET.ParseError as e:

        print(f"상세 법령 XML 파싱 에러: {e}")
        return []


    # --------------------------------------------------
    # 4. 조문 단위 Document 생성
    # --------------------------------------------------

    documents = []

    for article in detail_root.iter("조문단위"):

        # 실제 조문인지 확인
        article_type = article.findtext(
            "조문여부",
            default=""
        ).strip()

        if article_type and article_type != "조문":
            continue


        # 기본 조문 번호
        art_num = article.findtext(
            "조문번호",
            default=""
        ).strip()


        # 가지번호
        # 예: 제155조의2 → 조문번호 155 / 가지번호 2
        art_branch = article.findtext(
            "조문가지번호",
            default=""
        ).strip()


        # 제목
        art_title = article.findtext(
            "조문제목",
            default=""
        ).strip()


        # 조문 본문
        art_content = article.findtext(
            "조문내용",
            default=""
        ).strip()


        if not art_num:
            continue


        # --------------------------------------------------
        # 정확한 조문번호 생성
        # --------------------------------------------------

        if art_branch and art_branch != "0":

            full_art_num = (
                f"{art_num}의{art_branch}"
            )

        else:

            full_art_num = art_num


        # --------------------------------------------------
        # 항 내용
        # --------------------------------------------------

        sub_contents = []

        for paragraph in article.iter("항"):

            paragraph_text = paragraph.findtext(
                "항내용",
                default=""
            ).strip()

            if paragraph_text:
                sub_contents.append(
                    paragraph_text
                )


        # --------------------------------------------------
        # 검색에 사용할 조문 전체 텍스트
        # --------------------------------------------------

        text_parts = []

        if art_title:

            text_parts.append(
                f"[제{full_art_num}조 "
                f"({art_title})]"
            )

        else:

            text_parts.append(
                f"[제{full_art_num}조]"
            )


        if art_content:
            text_parts.append(art_content)


        text_parts.extend(sub_contents)


        full_text = "\n".join(text_parts).strip()


        if not full_text:
            continue


        # --------------------------------------------------
        # LangChain Document
        # --------------------------------------------------

        doc = Document(

            page_content=full_text,

            metadata={
                "law_name": target_law_name,
                "article_num": full_art_num,
                "article_title": art_title,
                "mst": law_id
            }

        )

        documents.append(doc)


    # --------------------------------------------------
    # 확인
    # --------------------------------------------------

    print(
        f"총 {len(documents)}개의 "
        "조문을 정상적으로 로드했습니다."
    )


    # 가지조문이 제대로 들어갔는지 일부 확인
    branch_articles = [
        doc.metadata["article_num"]
        for doc in documents
        if "의" in doc.metadata["article_num"]
    ]

    print(
        f"가지조문 수: {len(branch_articles)}"
    )

    print(
        "가지조문 예시:",
        branch_articles[:10]
    )


    return documents


# --------------------------------------------------
# 직접 실행 테스트
# --------------------------------------------------

if __name__ == "__main__":

    docs = fetch_law_data("소득세법")

    print("\n=== 로드 결과 확인 ===")

    for doc in docs[:5]:

        print(
            doc.metadata["article_num"],
            doc.metadata["article_title"]
        )