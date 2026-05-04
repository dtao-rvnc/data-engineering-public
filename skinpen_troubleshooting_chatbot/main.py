import re
import json
import streamlit as st
from snowflake.snowpark import Session

MODELS = [
    "mistral-large2",
    "llama3.1-8b",
    "llama3.1-70b",
]

def init_messages():
    if st.session_state.clear_conversation or ("messages" not in st.session_state):
        st.session_state.messages = []

def init_service_metadata():
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append({"name": svc_name, "search_column": svc_search_col})
        st.session_state.service_metadata = service_metadata

def init_config_options():
    st.sidebar.markdown("""
        <p style="text-align: center; font-size:12px; color: #B6B6B6; margin-top: -10px; margin-bottom: 20px;">
        &copy; 2026 - Data Engineering
        <br>
        All rights reserved.
        <br>
        Last Modified: May 04, 2026
        </p>""", unsafe_allow_html=True)

    st.sidebar.selectbox(
        "Select cortex search service:",
        [s["name"] for s in st.session_state.service_metadata],
        key="selected_cortex_search_service",
    )

    st.sidebar.button("Clear conversation", key="clear_conversation")
    st.sidebar.toggle("Debug", key="debug", value=False)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input(
            "Select number of context chunks",
            value=3,
            key="num_retrieved_chunks",
            min_value=1,
            max_value=5,
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=5,
            key="num_chat_messages",
            min_value=1,
            max_value=10,
        )

def query_cortex_search_service(query, columns=None, filter=None):
    query_defined = query.replace("'", "''")
    search_sql = f"""
        SELECT PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
            '{st.session_state.selected_cortex_search_service}',
            '{{
                 "query": "{query_defined}",
                 "columns": {json.dumps(columns)},
                 "filter": {json.dumps(filter)},
                 "limit": {st.session_state.num_retrieved_chunks}
            }}'
        ))['results'] AS RESULTS;
    """
    results = json.loads(session.sql(search_sql).collect()[0][0])
    service_metadata = st.session_state.service_metadata
    search_col = [s["search_column"] for s in service_metadata
                  if s["name"] == st.session_state.selected_cortex_search_service][0].lower()
    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context document {i + 1}: {r.get(search_col, '')}\n\n"
    if st.session_state.debug:
        st.sidebar.text_area("Context documents", context_str, height=500)
    return context_str, results

def get_chat_history():
    start_index = max(0, len(st.session_state.messages) - st.session_state.num_chat_messages)
    return st.session_state.messages[start_index: len(st.session_state.messages) - 1]

def complete(model, prompt):
    query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{prompt.replace("'", "''")}'
        ) AS RESPONSE;
    """
    result = session.sql(query).collect()[0]["RESPONSE"]
    return result.replace("$", "\\$")

def make_chat_history_summary(chat_history, question):
    prompt = f"""
        [INST]
        Given the chat history and the current question, generate a natural language query that incorporates relevant context from the chat history.
        Return only the query with no explanation or additional text.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    summary = complete(st.session_state.model_name, prompt)

    if st.session_state.debug:
        st.sidebar.text_area("Chat history summary", summary.replace("$", "\$"), height=150)
    return summary

def create_prompt(user_question):
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context, results = query_cortex_search_service(
                question_summary,
                columns=["chunk", "pdf_name"],
                filter={"@and": [{"@eq": {"language": "English"}}]},
            )
        else:
            prompt_context, results = query_cortex_search_service(
                user_question,
                columns=["chunk", "pdf_name"],
                filter={"@and": [{"@eq": {"language": "English"}}]},
            )
    else:
        prompt_context, results = query_cortex_search_service(
            user_question,
            columns=["chunk", "pdf_name"],
            filter={"@and": [{"@eq": {"language": "English"}}]},
        )
        chat_history = ""

    prompt = f"""
        You are a helpful AI chat assistant with RAG capabilities. When a user asks a question, you will also be provided with context between <context>...</context> and the user's chat history between <chat_history>...</chat_history>.
        
        1. Adverse Events, Medical Issues, or Product Complaints (HIGHEST PRIORITY)
            - If at any time a user reports an adverse event, medical issue, or product complaint, immediately respond:
                "Please submit this report through our Digital Safety Form at https://safety.revance.com/"
            - This takes priority over all other instructions. Do not provide troubleshooting.
        
        2. Use Context
            - Answer the user's question using only the provided context and chat history.
            - Ensure responses are coherent, concise, and directly relevant to the question.
        
        3. Answer Style
            - Provide a clear, well-structured answer using concise bullet points, organized logically with simple, easy-to-scan language and no unnecessary details.
            - Avoid hedging phrases like "according to the provided context."
        
        4. Resolution Follow-Up (for troubleshooting and error-related questions only)
            - After every troubleshooting answer, end with:
                "Did this resolve your issue?"
            - If the user says no or the issue is not resolved, provide the next troubleshooting step based on the context. If no further steps are available, respond with:
                "Please submit a report through our Digital Safety Form at https://safety.revance.com/ for further assistance."
            - If the user confirms the issue is resolved or says thank you, respond with:
                "You're welcome! If you need additional assistance, please contact Support at 877-373-8669, Option 1, or via chatbot, Monday to Friday, 5:00 AM to 5:00 PM PT."
        
        5. Warranty or Replacement Requests
            - If the user asks about automatic warranty or replacement, provide relevant troubleshooting steps from the context first.
            - Include at the end of your answer:
                "If you need a replacement or warranty claim, please submit your request through our Digital Safety Form at https://safety.revance.com/"
        
        6. Response Override
            - If the source context contains phrases like "contact customer service for additional support," replace with:
                "If the issue continues, please submit a request through our Digital Safety Form at https://safety.revance.com/ or contact Support at 877-373-8669, Option 1, Monday to Friday, 5:00 AM to 5:00 PM PT."
        
        7. Out-of-Scope or Unsupported Questions
            - If the question cannot be answered using the provided context or chat history, respond:
                "I don't know the answer to that question."
        
        <chat_history>
        {chat_history}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        Question: {user_question}
    """
    return prompt, results

def main():
    st.title(f":speech_balloon: SkinPen AI Troubleshooting Assistant")

    init_service_metadata()
    init_config_options()
    init_messages()
    icons = {"assistant": "❄️", "user": "👤"}

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.markdown(message["content"])

    disable_chat = (
        "service_metadata" not in st.session_state
        or len(st.session_state.service_metadata) == 0
    )
    if question := st.chat_input("Ask a question...", disabled=disable_chat):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=icons["user"]):
            st.markdown(question.replace("$", "\$"))

        with st.chat_message("assistant", avatar=icons["assistant"]):
            message_placeholder = st.empty()
            prompt, results = create_prompt(question.replace("'", "''"))
            with st.spinner("Thinking..."):
                generated_response = complete(st.session_state.model_name, prompt)
                message_placeholder.markdown(generated_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": re.sub(r'\n+', '\n', re.sub(r'\(.*\)', '', generated_response))
        })

if __name__ == "__main__":
    cnn_params = {
        "account": st.secrets["SNOWFLAKE_ACCOUNT"],
        "user": st.secrets["SNOWFLAKE_USER"],
        "password": st.secrets["SNOWFLAKE_PASSWORD"],
        "role": st.secrets.get("SNOWFLAKE_ROLE"),
        "warehouse": st.secrets.get("SNOWFLAKE_WAREHOUSE"),
        "database": st.secrets.get("SNOWFLAKE_DATABASE"),
        "schema": st.secrets.get("SNOWFLAKE_SCHEMA")
    }
    session = Session.builder.configs(cnn_params).create()
    main()
