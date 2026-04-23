import traceback
import tiktoken
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from urllib.parse import urlparse


SUMMARIZE_PROMPT = """다음 유튜브 콘텐츠 내용을 아래와 같이 한국어 마크다운 형식으로 요약해주세요.

### 콘텐츠
{content}

### 요약할 작업
- 전체 내용 요약(1000자):
- **부동산** 관련 의견요약(상승인지 하강인지):
- **고급 자동차** 보유 내용 요약(차종도 표시):
- **명품** 보유 과시 요약(명품 브랜드 표시):
- **후원금** 홍보 내용 요약(계좌번호 표시):
"""


# --------------------------------------------------
# 페이지 초기화
# --------------------------------------------------
def init_page():
    st.set_page_config(
        page_title="유튜브 채널 요약하기",
        page_icon="♣",
        layout="wide",
    )

    st.header("유튜브 요약하기 ♧")
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
        model = st.sidebar.radio("Choose Anthropic Model", models)

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
# 요약 체인
# --------------------------------------------------
def init_summarize_chain(
    provider,
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
            ("user", SUMMARIZE_PROMPT),
        ]
    )

    output_parser = StrOutputParser()

    return prompt | llm | output_parser


# --------------------------------------------------
# 전체 체인
# --------------------------------------------------
def init_chain(
    provider,
    openai_api_key="",
    anthropic_api_key="",
    google_api_key="",
):
    summarize_chain = init_summarize_chain(
        provider=provider,
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

    chain = RunnableLambda(route)
    return chain


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
# 유튜브 내용 가져오기
# --------------------------------------------------
def get_content_utube(url):
    with st.spinner("Fetching Youtube..."):
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=["ko", "en"],
            )

            res = loader.load()

            if res:
                return res[0].page_content
            return None

        except Exception as e:
            st.error(f"Error occurred: {e}")
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

    if url := st.text_input("YouTube URL 입력", key="input"):
        is_valid_url = validate_url(url)

        if not is_valid_url:
            st.error("유효한 URL을 입력해주세요.")
            return

        if content := get_content_utube(url):
            chain = init_chain(
                provider=provider,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                google_api_key=google_api_key,
            )

            st.markdown("## Summary")
            st.write_stream(
                chain.stream({"content": content})
            )

            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)


if __name__ == "__main__":
    main()