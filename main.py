from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_chroma import Chroma
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# os.environ['OPENAI_API_KEY']= YOUR OPEN API KEY
# os.environ['OPENAI_BASE_URL'] = YOUR OPEN API BASE URL

loader = PyPDFLoader("./contracts/SampleContract-Shuttle.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
all_splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What is data about?")
print(len(retrieved_docs))

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
print(example_messages)



def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What are the parties involved?"):
    print(chunk, end="", flush=True)
