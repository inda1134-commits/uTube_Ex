import traceback
import tiktoken
import streamlit as st
import requests

from bs4 import BeautifulSoup

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# YoutubeLoader 제거
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter

from urllib.parse import urlparse, parse_qs


# --------------------------------------------------
# 프롬프트
# --------------------------------------------------
YOUTUBE_SUMMARIZE_PROMPT = """다음 유튜브 콘텐츠 내용을 아래와 같이 한국어 마크다운 형식으로 요약해주세요.

### 콘텐츠
{content}

### 요약할 작업
- 전체 내용 요약(500자):
- 핵심 주장 또는 핵심 메시지:
- 중요한 숫자/통계/가격 정보:
- 주의해야 할 내용:
----
- **부동산** 관련 의견요약(상승인지 하강인지):
- **고급 자동차** 보유 내용 요약(차종도 표시):
- **명품** 보유 과시 요약(명품 브랜드 표시):
- **후원금** 홍보 내용 요약(계좌번호 표시):
"""

WEBSITE_SUMMARIZE_PROMPT = """다음 웹사이트 본문 내용을 한국어 마크다운 형식으로 요약해주세요.

### 웹사이트 본문
{content}

### 요약할 작업
- 전체 내용 요약 (500자):
- 핵심 주장 또는 핵심 메시지:
- 중요한 숫자/통계/가격 정보:
- 주의해야 할 내용:
"""


# --------------------------------------------------
# 페이지 초기화
# --------------------------------------------------
def init_page():
    st.set_page_config(
        page_title="URL 콘텐츠 요약하기",
        page_icon="♣",
        layout="wide",
    )

    st.header("유튜브 / 웹사이트 요약하기 ♧")
    st.sidebar.title("LLM 설정")


# --------------------------------------------------
# API Key 입력 UI
# --------------------------------------------------
def input_api_keys():
    st.sidebar.markdown("## API Key 입력")

    provider = st.sidebar.selectbox(
        "사용할 LLM Provider 선택",
        (
            "OpenAI",
            "Anthropic",
            "Google Gemini",
        ),
    )

    openai_api_key = ""
    anthropic_api_key = ""
    google_api_key = ""

    if provider == "OpenAI":
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )

    elif provider == "Anthropic":
        anthropic_api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
        )

    elif provider == "Google Gemini":
        google_api_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            placeholder="AIza...",
        )

    return provider, openai_api_key, anthropic_api_key, google_api_key


# --------------------------------------------------
# 모델 선택
# --------------------------------------------------
def select_model(
    provider,
    openai_api_key="",
    anthropic_api_key="",
    google_api_key="",
    temperature=0,
):
    if provider == "OpenAI":
        models = ("gpt-5-mini", "gpt-5.2")
        model = st.sidebar.radio("Choose OpenAI Model", models)

        if not openai_api_key:
            st.warning("OpenAI API Key를 입력해주세요.")
            st.stop()

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openai_api_key,
        )

    elif provider == "Anthropic":
        models = ("claude-sonnet-4-5",)
        st.sidebar.radio("Choose Anthropic Model", models)

        if not anthropic_api_key:
            st.warning("Anthropic API Key를 입력해주세요.")
            st.stop()

        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=temperature,
            api_key=anthropic_api_key,
        )

    elif provider == "Google Gemini":
        models = ("gemini-2.5-flash",)
        model = st.sidebar.radio("Choose Gemini Model", models)

        if not google_api_key:
            st.warning("Google API Key를 입력해주세요.")
            st.stop()

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=google_api_key,
        )


# --------------------------------------------------
# URL 종류 판별
# --------------------------------------------------
def is_youtube_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    youtube_domains = [
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "m.youtube.com",
    ]

    return any(d in domain for d in youtube_domains)


# --------------------------------------------------
# YouTube video_id 추출
# --------------------------------------------------
def extract_youtube_video_id(url):
    try:
        parsed = urlparse(url)

        # youtu.be 형식
        if parsed.netloc in ["youtu.be"]:
            return parsed.path.lstrip("/")

        # youtube.com/watch?v=
        if "youtube.com" in parsed.netloc:
            query = parse_qs(parsed.query)

            if "v" in query:
                return query["v"][0]

            # shorts 지원
            if "/shorts/" in parsed.path:
                return parsed.path.split("/shorts/")[1].split("/")[0]

            # embed 지원
            if "/embed/" in parsed.path:
                return parsed.path.split("/embed/")[1].split("/")[0]

        return None

    except Exception:
        return None


# --------------------------------------------------
# 요약 체인
# --------------------------------------------------
def init_summarize_chain(
    provider,
    prompt_text,
    openai_api_key="",
    anthropic_api_key="",
    google_api_key="",
):
    llm = select_model(
        provider=provider,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        google_api_key=google_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", prompt_text),
        ]
    )

    output_parser = StrOutputParser()

    return prompt | llm | output_parser


# --------------------------------------------------
# 전체 체인
# --------------------------------------------------
def init_chain(
    provider,
    prompt_text,
    openai_api_key="",
    anthropic_api_key="",
    google_api_key="",
):
    summarize_chain = init_summarize_chain(
        provider=provider,
        prompt_text=prompt_text,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        google_api_key=google_api_key,
    )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-5",
        chunk_size=16000,
        chunk_overlap=0,
    )

    text_split = RunnableLambda(
        lambda x: [
            {"content": doc}
            for doc in text_splitter.split_text(x["content"])
        ]
    )

    text_concat = RunnableLambda(
        lambda x: {"content": "\n".join(x)}
    )

    map_reduce_chain = (
        text_split
        | summarize_chain.map()
        | text_concat
        | summarize_chain
    )

    def route(x):
        try:
            encoding = tiktoken.encoding_for_model("gpt-5")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        token_count = len(encoding.encode(x["content"]))

        if token_count > 16000:
            return map_reduce_chain

        return summarize_chain

    return RunnableLambda(route)


# --------------------------------------------------
# URL 검증
# --------------------------------------------------
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# --------------------------------------------------
# 유튜브 내용 가져오기 (수정 완료)
# --------------------------------------------------
def get_content_youtube(url):
    with st.spinner("Fetching YouTube..."):
        try:
            video_id = extract_youtube_video_id(url)

            if not video_id:
                st.error("YouTube video_id를 추출할 수 없습니다.")
                return None

            transcript = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=["ko", "en"],
            )

            if not transcript:
                st.error("자막을 가져오지 못했습니다.")
                return None

            content = "\n".join(
                item["text"]
                for item in transcript
                if item.get("text")
            )

            if not content.strip():
                st.error("자막 내용이 비어 있습니다.")
                return None

            return content[:50000]

        except Exception as e:
            st.error(f"YouTube 처리 오류: {e}")
            st.code(traceback.format_exc())
            return None


# --------------------------------------------------
# 웹사이트 본문 가져오기
# --------------------------------------------------
def get_content_website(url):
    with st.spinner("Fetching Website..."):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }

            response = requests.get(
                url,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            # 불필요 태그 제거
            for tag in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "header",
                    "aside",
                    "noscript",
                ]
            ):
                tag.decompose()

            text = soup.get_text(separator="\n")

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            cleaned_text = "\n".join(lines)

            if len(cleaned_text) < 100:
                return None

            return cleaned_text[:50000]

        except Exception as e:
            st.error(f"웹사이트 처리 오류: {e}")
            st.code(traceback.format_exc())
            return None


# --------------------------------------------------
# 테스트용 코드 (유지)
# --------------------------------------------------
prompt = PromptTemplate.from_template("Say: {content}")


def to_upper(x):
    return {"content": x["content"].upper()}


to_upper_chain = RunnableLambda(to_upper) | prompt


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    init_page()

    (
        provider,
        openai_api_key,
        anthropic_api_key,
        google_api_key,
    ) = input_api_keys()

    if url := st.text_input(
        "URL 입력 (YouTube 또는 웹사이트)",
        key="input",
    ):
        if not validate_url(url):
            st.error("유효한 URL을 입력해주세요.")
            return

        # URL 종류 판별
        if is_youtube_url(url):
            content = get_content_youtube(url)
            prompt_text = YOUTUBE_SUMMARIZE_PROMPT
            content_type = "YouTube"

        else:
            content = get_content_website(url)
            prompt_text = WEBSITE_SUMMARIZE_PROMPT
            content_type = "Website"

        if not content:
            st.error("콘텐츠를 가져오지 못했습니다.")
            return

        chain = init_chain(
            provider=provider,
            prompt_text=prompt_text,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key,
        )

        st.markdown(f"## {content_type} Summary")
        st.write_stream(
            chain.stream({"content": content})
        )

        st.markdown("---")
        st.markdown("## Original Text")
        st.write(content)


if __name__ == "__main__":
    main()