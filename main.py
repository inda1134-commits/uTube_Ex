import traceback
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

SUMMARIZE_PROMPT = """다음 콘텐츠 내용을 약 300자 정도로 알기 쉽게 요약해주세요.
======
{context}
======
한국어로 작성해 주세요!
"""

def init_page():
    st.set_page_config(page_title="웹사이트 요약하기", page_icon="♣")
    st.header("웹사이트 요약하기 ♧")
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
    
def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([("user", SUMMARIZE_PROMPT)])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain

def validate_url(url):
    """URL이 유효한지 판단하는 함수"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_content(url):
    try:
        with st.spinner("Fetching Website ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write(traceback.format_exc())
        return None

def main():
    init_page()
    chain=init_chain()
    
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("Please input valid url")
        else:
            if context := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"context": context}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(context)

if __name__ == "__main__":
    main()