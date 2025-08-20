import streamlit as st
import json
import os

USER_DB_FILE = "user_db.json"

def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as file:
            return json.load(file)
    return {}

def save_user_db(user_db):
    with open(USER_DB_FILE, "w") as file:
        json.dump(user_db, file)

def login_ui():
    st.title("Jubile Chatbot Login")

    user_db = load_user_db()

    option = st.radio("Select an option", ("Login", "Register"))

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username in user_db and user_db[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.current_page = "law" 
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    elif option == "Register":
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        register_btn = st.button("Register")

        if register_btn:
            if new_username in user_db:
                st.error("Username already exists. Please choose a different username.")
            elif not new_username or not new_password:
                st.error("Both fields are required.")
            else:
                user_db[new_username] = new_password
                save_user_db(user_db)
                st.success("Registration successful! You can now log in.")
