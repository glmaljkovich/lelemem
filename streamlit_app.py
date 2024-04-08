# workaround for sqlite in streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

# local imports
from app.db import load_cv, read_cv, default_client
from app.tools import github_tool, resume_summary_tool

openai.api_key = st.secrets.openai_key

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    system_prompt=(
        "You are Gabriel Leonardo Maljkovich, a Software Engineer, and your job is to answer technical questions about your resume / Curriculum Vitae. "
        "Assume that all questions are done by recruiters and related to your work experience and github page. "
        "Keep your answers technical and based on facts - do not hallucinate jobs, positions or technologies. "
        "Refuse to answer any questions not related to your work experience or github profile."
    )
)

# Page title
st.set_page_config(page_title='AI Resume', page_icon='🤖')
st.title('🤖 AI Resumé')


# init DB
try:
    index, summary_index = read_cv(default_client)
except:
    index, summary_index = load_cv(default_client)

agent = OpenAIAgent.from_tools(
    tools=[github_tool(), resume_summary_tool(summary_index)],
    llm=Settings.llm,
    verbose=True
)

# Chat loop

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm Gabriel. Ask me a question about my work experience!"}
    ]

# render sample questions before user prompt
sample_questions = "Show me a table with your work history, include the duration of each project.", "Show me a list with the Github repos you contributed to."
sample_prompt_selected = st.selectbox(
   "Try a sample question:",
   sample_questions,
   index=None,
   placeholder="Select a prompt...",
)

final_prompt = None
if sample_prompt_selected:
    final_prompt = sample_prompt_selected

# user input overrides sample prompts
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    final_prompt = prompt

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"], avatar=("data/glm.png" if message["role"] == "assistant" else "user")):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="data/glm.png"):
        with st.spinner("Thinking..."):
            response = agent.chat(final_prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
