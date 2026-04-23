import traceback
import tiktoken
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import YoutubeLoader

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
# import os
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

SUMMARIZE_PROMPT = """다음 유튜브 콘텐츠 내용을 아래와 같이 한국어 마크다운 형식으로 요약해주세요...
### 콘텐츠
{content}
### 요약할 작업
- 전체 내용 요약(1000자):
- **부동산** 관련 의견요약(상승인지 하강인지):
- **고급 자동차** 보유 내용 요약(차종도 표시):
- **명품** 보유 과시 요약(명품 브랜드 표시):
- **후원금** 홍보 내용 요약(계좌번호 표시):
"""

def init_page():
    st.set_page_config(page_title="유튜브 채널 요약하기", page_icon="♣")
    st.header("유튜브 요약하기 ♧")
    st.sidebar.title("Options")

def select_model(temperature=0):
    models = ("GPt-5 mini", "GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)
    if model=="GPt-5 mini":
        return ChatOpenAI(temperature=temperature, model="gpt-5-mini")
    elif model=="GPT-5.2":
        return ChatOpenAI(tempereture=temperature, model="gpt-5.2")
    elif model=="Claude Sonnet 4.5":
        return ChatAnthropic(tempereture=temperature, model="claude-sonnet-4-5-20250929")
    elif model=="Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(tempereture=temperature, model="gemini-2.5-flash")

def init_summarize_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", SUMMARIZE_PROMPT),
        ]
    )
    output_paraser = StrOutputParser()
    return prompt | llm | output_paraser
    
def init_chain():
    summarize_chain = init_summarize_chain()
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-5",
        chunk_size = 16000,
        chunk_overlap=0,
    )
    text_split = RunnableLambda(
        lambda x: [{"content": doc} for doc in text_splitter.split_text(x["content"])]
    )
    
    text_concat = RunnableLambda(lambda x: {"content": "\n".join(x)})
    map_reduce_chain = (
        text_split | summarize_chain.map() | text_concat | summarize_chain
    )
    
    def route(x):
        encoding = tiktoken.encoding_for_model("gpt-5")
        token_count = len(encoding.encode(x["content"]))
        if token_count > 16000:
            return map_reduce_chain
        else:
            return summarize_chain
        
    chain = RunnableLambda(route)
    return chain
    
def validate_url(url):
    """URL이 유효한지 판단하는 함수"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_content_utube(url):
    with st.spinner("Fetching Youtube ..."):
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=["ko", "en"],
            )
            res = loader.load()
            if res:
                return res[0].page_content
            else:
                return None
        except Exception as e:
            st.error(f"Error occured: {e}")
            st.write(traceback.format_exc())
            return None

prompt = PromptTemplate.from_template("Say: {content}")

def to_upper(x):
    return {"content": x["content"].upper()}

to_uppper = RunnableLambda(to_upper)
to_upper_chain = to_upper | prompt
# print(to_upper_chain.invoke({"content": "yeah!"}))

def main():
    init_page()
    chain=init_chain()
    
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("Please input valid url")
        else:
            if content := get_content_utube(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

if __name__ == "__main__": 
    main()