import os
import sys
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message as st_message
import time
import pandas as pd

# own classes
from scrap.scrapper import WebScrapper
from rag.summarization import WebSummarizer
from rag.ingest import EmbeddingIngestor
from rag.chatbot import ChatBot
from config.ai_models import list_models
from couch_db.couchdb import couchbase_cnn

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
nest_asyncio.apply()

# Session variables
if "url_submitted" not in st.session_state:
    st.session_state.url_submitted = False
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------------------
# Page Config in streamlit
# ---------------------------

st.set_page_config(layout="wide", page_title="Web-ChatBot")
st.title("Project Chatbot with Multiple LLMs + RAG")

# ---------------------------
# Streamlit UI
# ---------------------------

page = st.sidebar.selectbox("Menu", ["Home", "AI Chatbot", "Reports", "Benchmarks"])

if page == "Home":
    st.markdown(
        """
        ## Welcome to Web-Chatbot
        **Web-Chatbot** is a small chatbot empowered by integration of multiple LLM with RAG of website knowledge extraction through LangChain.
        
        **Functionalities:**
        - **Web Scraping:** Crawl and extract web page content.
        - **Web Summarization:** Generate detailed summaries of the extracted content.
        - **Create Embeddings:** Embeddings with FAISS for vector representation and retrieval of web-scraped information
        - **Chatbot Interface:** Execute Question-Answering task via a conversational agent.

        **Technologies:**
        - **LLM:** Models 
            1) deepseek-r1:1.5b
            2) qwen3:1.7b
            3) llama3.2:3b
            4) gemma2:2b
            5) gpt-4o
        - **FAISS:** vector database to store embeddings
        - **CouchDB:** document database to store the chatbot results
        - **LangChain:** framework to integrate LLM, external data and tools
        - **Streamlit:** python library to fast prototype web apps
        - **Craw4AI:** python library to crawl and scrape web pages
        
        Get started!
        """
    )    

elif page == "AI Chatbot":
    with st.form("url_form"):
        url_input = st.text_input("Enter a URL to crawl:")
        submit_url = st.form_submit_button("Submit URL")

        if submit_url and url_input:
            st.session_state.url_submitted = True
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""
    
    if st.session_state.url_submitted:
        col1, col2 = st.columns(2)

        with col1:
            st.header("1. Web-Scrapping")

            if not st.session_state.extraction_done:
                with st.spinner("Extracting website..."):
                    scraper = WebScrapper()
                    extracted = asyncio.run(scraper.crawl(url_input))
                    st.session_state.extracted_text = extracted
                    st.session_state.extraction_done = True
                st.success("Extraction complete!")

            preview = "\n".join([line for line in st.session_state.extracted_text.splitlines() if line.strip()][:5])
            st.text_area("Extracted Text Preview", preview, height=150)

            st.download_button(
                label="Download Extracted Text",
                data=st.session_state.extracted_text,
                file_name="extract_text.txt",
                mime="text/plain",
            )

            st.markdown("---")

            st.header("2. Web-Summarization")

            if st.button("Summarize Web Page", key="summarize_button"):
                with st.spinner("Summarizing..."):
                    summarizer = WebSummarizer()
                    st.session_state.summary = summarizer.summarize(st.session_state.extracted_text)
                st.success("Summarization complete!")

            if st.session_state.summary:
                st.subheader("Summarized Output")
                st.markdown(st.session_state.summary, unsafe_allow_html=False)

        with col2:
            st.header("3. Create Embeddings")

            if st.session_state.extraction_done and not st.session_state.embedding_done:
                if st.button("Create Embeddings"):
                    with st.spinner("Creating embeddings..."):
                        embeddings = EmbeddingIngestor()
                        st.session_state.vectorstore = embeddings.create_embeddings(st.session_state.extracted_text)
                        st.session_state.embedding_done = True
                    st.success("Vectors are created!")

            elif st.session_state.embedding_done:
                st.info("Embeddings have been created.")

            st.markdown("---")

            st.header("4. ChatBot")

            if st.session_state.embedding_done:
                
                selected_model = st.selectbox(
                    label = "--- Select LLM model ---",
                    options = list_models(),
                    index = 0,
                    help = "Select the LLM model for Chatbot"
                )

                chatbot = ChatBot(st.session_state.vectorstore, selected_model)
                user_input = st.text_input("Your Message:", key="chat_input")                

                if st.button("Send", key="send_button") and user_input:
                    # start chatbot
                    start_time = time.time()
                    
                    bot_answer = chatbot.qa(user_input)
                    
                    end_time = time.time()
                    total_time = (end_time - start_time)/60

                    st.session_state.chat_history.append({
                        "user": user_input, 
                        "bot": bot_answer, 
                        "time": total_time
                    })

                    chat_file_content = "\n\n".join([f"User: {chat['user']}\nBot: {chat['bot']}\nTime: {chat.get('time', 0):.2f} min" for chat in st.session_state.chat_history])
                    with open("history/chat_history.txt", "w", encoding="utf-8") as cf:
                        cf.write(chat_file_content)

                # show response in frontend
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        # print chatbot results in frontend
                        st.markdown(f"Time: {chat.get('time', 0):.2f} minutes")
                        st_message(chat["user"], is_user=True)
                        st_message(chat["bot"], is_user=False)

                        try:                            
                            # insert results chatbot in couchbase
                            success = couchbase_cnn.insert(
                                model_name = selected_model,
                                question = chat["user"],
                                answer = chat["bot"],
                                time = chat.get('time', 0),
                                score = None
                            )

                            if not success:
                                st.warning("Failed to save QA in couchbase!")
                        except Exception as e:
                            st.error(f"Couchbase error: {str(e)}")
            else:
                st.info("Please create embeddings to activate the chat.")

elif page == "Reports":
    
    # read data from couchbase
    list_docs = couchbase_cnn.read_documents()

    if list_docs is None:
        st.warning("No data is found in Couchbase!")

    # data transformation
    df = pd.DataFrame({
        'Model': list_docs['model_name'],
        'Question': list_docs['question'],
        'Answer': list_docs['answer'],
        'Time': list_docs['time'],
        'Score': list_docs['score']
    })
    
    df['Time'] = df['Time'].apply(lambda x: f"{x:.2f}")
    df['Score'] = df['Score'].fillna(0)
    df['Actions'] = "View"
    df['row_id'] = range(len(df))    

    # datatable with pagination
    st.subheader("Historical Report of Conversations")
    
    # pagination
    page_size = 5
    page_num = st.number_input("Page number", min_value = 1, 
                                max_value = max(1, len(df)-1)//page_size + 1, value = 1)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    # display the current page
    selected_row = st.dataframe(data = df.iloc[start_idx:end_idx],
                 column_config = {
                     "Question": st.column_config.TextColumn(
                         "Question", width = "medium", help = "Click row to expand"
                     ),
                     "Answer": st.column_config.TextColumn(
                         "Answer", width = "medium", help = "Click row to expand"
                     ),
                     "Actions": st.column_config.TextColumn(
                         "Actions", width = "small"
                     ),
                     "row_id": None
                 }, use_container_width = True)
    
    if 'selected_row_id' not in st.session_state:
        st.session_state.selected_row_id = None

    # create form for each view button
    for idx, row in df.iloc[start_idx:end_idx].iterrows():
        cols = st.columns([5, 1])
        with cols[1]:
            if st.button("View", key = f"view_{row['row_id']}"):
                st.session_state.selected_row_id = row['row_id']                
    
    # display the detailed view
    if st.session_state.selected_row_id is not None:
        selected_data = df[df['row_id'] == st.session_state.selected_row_id].iloc[0]

        with st.container():
            st.markdown("---")
            st.subheader("Conversation Details")

            with st.container():
                cols = st.columns(3)
                
                with cols[0]:
                    st.markdown("**Model Used**")
                    st.info(selected_data['Model'])
                with cols[1]:
                    st.markdown("**Time (min)**")
                    st.info(selected_data['Time'])
                with cols[2]:
                    st.markdown("**Score**")
                    st.info(selected_data['Score'])
            
            with st.container():
                st.markdown("**Question**")
                st.success(selected_data['Question'])
            
            with st.container():
                st.markdown("**Answer**")
                st.text_area("Full Answer",
                             value = selected_data['Answer'],
                             height = 450,
                             disabled = False,
                             key = f"answer_{selected_data['row_id']}")
                
            if st.button("Close details"):
                st.session_state.selected_row_id = None
