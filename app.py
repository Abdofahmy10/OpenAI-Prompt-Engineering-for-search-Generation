import os
from api_key import api_key
import streamlit as st 
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain.chains import LLMChain



os.environ['OPENAI_API_KEY'] = api_key


st.title(' Creating Research  🔥 ⚡ ')
prompt = st.text_input('Please Enter your subject ')


title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='suggest to me the best Title about {topic}'
)

inforamation_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write to  me information based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# LLM Models 
Model = OpenAI(temperature=0.8) 
title_chain = LLMChain(llm=Model, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=Model, prompt=inforamation_template, verbose=True, output_key='script', memory=script_memory)
wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write('Here is your Research ... ') 
   # st.write("Your Information : \n ",script) 

    with st.expander('suggestion Title...🔗'): 
        st.info(title_memory.buffer)

    with st.expander('Information ...🚀'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research ...⚡ '): 
        st.info(wiki_research)