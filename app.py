import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_templates import css, bot_template, user_template
import openai
import speech_recognition as sr
from PIL import Image
import io

def get_pdf_text(pdf_docs):
    text = "" 
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text       

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks                          

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore        

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def handle_gpt4_chat(user_question):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": user_question}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_quiz(text_chunks):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    prompt = "Generate a quiz based on the following text:\n\n" + "\n".join(text_chunks)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1500
    )
    return response.choices[0].text

def describe_image(image_bytes):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.Image.create(
        file=image_bytes,
        model="image-alpha-001"
    )
    return response['choices'][0]['text']

def main():
    load_dotenv()
    st.set_page_config(page_title="STUDY PAL",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF:books:")

    # PDF Chat Section
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz..."):
            quiz = generate_quiz(st.session_state.text_chunks)
            st.write(quiz)

    # GPT-4 Chat Section
    st.subheader("Chat with GPT-4")
    gpt4_question = st.text_input("Ask GPT-4 a question:")
    if gpt4_question:
        gpt4_response = handle_gpt4_chat(gpt4_question)
        st.write(bot_template.replace("{{MSG}}", gpt4_response), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.session_state.text_chunks = text_chunks

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    # Image Upload Section
    st.subheader("Image Recognition")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        description = describe_image(image_bytes.read())
        st.write(bot_template.replace("{{MSG}}", description), unsafe_allow_html=True)

    # Voice Recognition Section
    st.subheader("Voice Recognition")
    if st.button("Record Voice"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Recording...")
            audio = recognizer.listen(source)
            st.write("Done recording.")
            try:
                voice_text = recognizer.recognize_google(audio)
                st.write(f"Recognized Text: {voice_text}")
                gpt4_response = handle_gpt4_chat(voice_text)
                st.write(bot_template.replace("{{MSG}}", gpt4_response), unsafe_allow_html=True)
            except sr.UnknownValueError:
                st.write("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == '__main__':
    main()
