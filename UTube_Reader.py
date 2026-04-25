import traceback
import tiktoken
import streamlit as st
import requests
import os
import tempfile
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse, parse_qs
from openai import OpenAI
import yt_dlp

# --- 프롬프트 설정 ---
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

# --- 페이지 초기화 및 UI ---
def init_page():
    st.set_page_config(page_title="URL 콘텐츠 요약하기", page_icon="♣", layout="wide")
    st.header("유튜브 / 웹사이트 요약하기 ♧")

def input_api_keys():
    st.sidebar.markdown("## API Key 입력")
    provider = st.sidebar.selectbox("사용할 LLM Provider 선택", ("OpenAI", "Anthropic", "Google Gemini"))
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password")
    google_key = st.sidebar.text_input("Google API Key", type="password")
    return provider, openai_key, anthropic_key, google_key

def select_model(provider, openai_key, anthropic_key, google_key):
    if provider == "OpenAI":
        if not openai_key: st.warning("OpenAI API Key 필요"); st.stop()
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    elif provider == "Anthropic":
        if not anthropic_key: st.warning("Anthropic API Key 필요"); st.stop()
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=anthropic_key)
    elif provider == "Google Gemini":
        if not google_key: st.warning("Google API Key 필요"); st.stop()
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key)

# --- 기능 함수 ---
def is_youtube_url(url):
    return any(d in urlparse(url).netloc for d in ["youtube.com", "youtu.be"])

def extract_youtube_video_id(url):
    parsed = urlparse(url)
    if parsed.netloc == "youtu.be": return parsed.path.lstrip("/")
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query: return query["v"][0]
        if "/shorts/" in parsed.path: return parsed.path.split("/shorts/")[1].split("/")[0]
    return None

def get_youtube_transcript(video_id):
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        return " ".join([x['text'] for x in srt])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def transcribe_youtube_audio(url, openai_api_key):
    if not openai_api_key:
        st.error("자막이 없어 Whisper를 실행해야 하지만 OpenAI API Key가 없습니다.")
        return None
    try:
        with st.status("오디오 추출 및 Whisper 변환 중..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = os.path.join(tmpdir, "audio.mp3")
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                client = OpenAI(api_key=openai_api_key)
                with open(audio_path, "rb") as f:
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
                return transcript.text
    except Exception as e:
        st.error(f"Whisper 변환 오류: {e}")
        return None

def get_website_content(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text()
    except:
        return "웹사이트 내용을 가져올 수 없습니다."

# --- 메인 실행 로직 ---
def main():
    init_page()
    provider, o_key, a_key, g_key = input_api_keys()
    
    url = st.text_input("요약할 URL을 입력하세요 (YouTube 또는 웹사이트)")
    
    if st.button("요약하기"):
        if not url:
            st.warning("URL을 입력해주세요.")
            return

        content = ""
        prompt_template = WEBSITE_SUMMARIZE_PROMPT

        if is_youtube_url(url):
            prompt_template = YOUTUBE_SUMMARIZE_PROMPT
            video_id = extract_youtube_video_id(url)
            if video_id:
                content = get_youtube_transcript(video_id)
                if not content:
                    content = transcribe_youtube_audio(url, o_key)
        else:
            content = get_website_content(url)

        if content:
            llm = select_model(provider, o_key, a_key, g_key)
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | StrOutputParser()
            
            with st.spinner("요약 중..."):
                # 텍스트가 너무 길 경우를 대비해 앞부분 10,000자만 사용 (필요시 Splitter 사용)
                result = chain.invoke({"content": content[:10000]})
                st.markdown("### 요약 결과")
                st.write(result)
        else:
            st.error("콘텐츠를 추출할 수 없습니다.")

if __name__ == "__main__":
    main()
