from dotenv import load_dotenv
import streamlit as st
import base64
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

prompt_tempelate="""
You are a doctor and have expertize in human anatomy.
Answer the following question based on the given context, which can include text, images and tables.
Context:{context}
Question:{question}
Elaborate the answer based on given context only.
Answer:"""

def get_vectorstore():
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    db=FAISS.load_local('./faiss_index',embeddings)
    return db

def get_chain():
    qa_chain=LLMChain(llm=ChatGoogleGenerativeAI(model='gemini-pro',convert_system_message_to_human=True), prompt=PromptTemplate.from_template(prompt_tempelate))
    return qa_chain

def generate_ans(question):
  db=st.session_state.vectorstore
  chain=st.session_state.chain
  relevant_docs=db.similarity_search(question)
  context=""
  relevant_images=[]
  for d in relevant_docs:
    if d.metadata['type']=='text':
      context+='[text]'+d.metadata['orignal_content']
    elif d.metadata['type']=='table':
      context+='[table]'+d.metadata['orignal_content']
    elif d.metadata['type']=='image':
      context+='[image]'+d.page_content
      relevant_images.append(d.metadata['orignal_content'])
  result=chain.run({'context':context,"question":question})
  st.write(result)
  if len(relevant_images)>0:
    st.image(base64.b64decode(relevant_images[0]))

def main():
    if "chain" not in st.session_state:
        st.session_state.chain=get_chain()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore=get_vectorstore()
    st.set_page_config(page_title="Multimodal RAG", page_icon="	:muscle:")
    st.header("Multimodal RAG - HUMAN ANATOMY :man-cartwheeling:")
    question=st.text_input("Ask anything related to human anatomy")
    if(question):
       generate_ans(question)
   

if __name__ =="__main__":
   load_dotenv()
   main()

