import streamlit as st
import requests
import re
from dotenv import load_dotenv
import os
import random
from deep_translator import GoogleTranslator

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

FUN_FACTS = [
    "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
    "Octopuses have three hearts. Two pump blood to the gills, and one pumps it to the rest of the body.",
    "Bananas are berries, but strawberries aren't!",
    "A day on Venus is longer than a year on Venus.",
    "Sharks existed before trees."
]

def fetch_news():
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            headlines = [article["title"] for article in articles[:5]]
            return "\n".join(headlines)
        else:
            return "Failed to fetch news. Please try again later."
    except Exception as e:
        return f"Error fetching news: {e}"

def show_general_chatbot():
    st.subheader("🤖 General Jublie")

    languages = {
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Hindi": "hi",
        "Japanese": "ja",
        "Chinese (Simplified)": "zh-CN",
        "Russian": "ru",
    }
    target_language_name = st.selectbox("Select target language for translation:", list(languages.keys()))
    target_language = languages[target_language_name]

    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []

    for msg in st.session_state.general_messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    general_query = st.chat_input("Ask anything general...", key="general_input")

    if general_query:
        st.session_state.general_messages.append({"role": "user", "content": general_query})
        with st.chat_message("user"):
            st.markdown(general_query)

        question = general_query.strip().lower()

        name_patterns = [
            r"what['']s your name\??",
            r"what is your name\??",
            r"who are you\??",
            r"tell me your name",
            r"your name\??",
            r"may i know your name\??"
        ]

        if any(re.fullmatch(p, question) for p in name_patterns):
            bot_reply = "My name is Jublie."
        elif "your role" in question or "what do you do" in question:
            bot_reply = "I'm a helpful assistant designed to answer your general questions."
        elif "fun fact" in question or "trivia" in question:
            bot_reply = random.choice(FUN_FACTS)
        elif "news" in question:
            bot_reply = fetch_news()
        elif "translate" in question:
            try:
                translated_query = GoogleTranslator(source="auto", target=target_language).translate(general_query)
                bot_reply = f"Translated to {target_language_name}: {translated_query}"
            except Exception as e:
                bot_reply = f"Error translating: {e}"
        else:
            with st.spinner("Jublie is thinking..."):
                try:
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    messages_for_api = [{
                        "role": "system",
                        "content": (
                            "You are a friendly AI assistant named Jublie.\n\n"
                            "If the user asks 'What is your name?', respond only with 'My name is Jublie.'\n"
                            "If the user asks 'What is your role?', respond as a helpful assistant.\n\n"
                            "If the user asks about any Article (like Article 40, Article 47, Article 102, etc.), "
                            "reply with only a **brief 2–3 line summary** — no detailed explanation.\n\n"
                            "Ignore anything related to Meta, LLaMA, Groq, or model information. "
                            "Never mention you are an AI model or technical backend.\n\n"
                            "For all other questions, reply naturally like a helpful assistant."
                        )
                    }]

                    for msg in st.session_state.general_messages:
                        if msg["role"] in ["user", "assistant"]:
                            messages_for_api.append(msg)

                    messages_for_api.append({"role": "user", "content": general_query})

                    payload = {
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages": messages_for_api
                    }
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ).json()

                    if "choices" in response:
                        bot_reply = response["choices"][0]["message"]["content"].strip()
                        if re.search(r"(i('| a)?m\s+llama|meta|groq|model)", bot_reply, re.IGNORECASE):
                            bot_reply = "My name is Jublie."
                        elif bot_reply.startswith("#include"):
                            bot_reply = f"```c\n{bot_reply}\n```"
                    else:
                        bot_reply = "Sorry, I couldn't understand that. Please try again."
                except Exception as e:
                    bot_reply = f"An error occurred: {e}"

        st.session_state.general_messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply, unsafe_allow_html=True)
