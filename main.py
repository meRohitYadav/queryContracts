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


os.environ['OPENAI_API_KEY']="eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJhZXNKN2kxNGNidnVuTU40MTJrOU5yZ2ROeENhTlJudTNPbC1TU08ycFlJIn0.eyJleHAiOjE3MTM3ODMwMDMsImlhdCI6MTcxMzc4MTIwNCwiYXV0aF90aW1lIjoxNzEzNzgxMjAzLCJqdGkiOiJhOTEwMjMzMi03YTU1LTRlNzUtYjFiMC02M2U0MDZmODM3ZTUiLCJpc3MiOiJodHRwczovL2F1dGgubWNraW5zZXkuaWQvYXV0aC9yZWFsbXMvciIsImF1ZCI6ImJjZDIzNzI4LTNkMjctNDQ3Yy1hMGE5LWVhY2FmMzkzYTZmNSIsInN1YiI6Ijg1OGI5YTVjLTAyMzktNGE4YS1iM2Q4LWU1ZjkwNGE4NGM1MSIsInR5cCI6IklEIiwiYXpwIjoiYmNkMjM3MjgtM2QyNy00NDdjLWEwYTktZWFjYWYzOTNhNmY1Iiwibm9uY2UiOiJoR091Uk5wWk5UUVp5dXdzX2RPb1g4VDVwVGowS2Z6UkdYd2VEZUlRTmdZIiwic2Vzc2lvbl9zdGF0ZSI6IjUxODc0M2RiLWFjZGItNDAwZC1iYTVkLWFiMTIwYWM4ODg3MyIsImF0X2hhc2giOiI3Y01zdGFoOU5udW80Y2xxbl9VMFNnIiwibmFtZSI6IlJvaGl0IFlhZGF2IiwiZ2l2ZW5fbmFtZSI6IlJvaGl0IiwiZmFtaWx5X25hbWUiOiJZYWRhdiIsInByZWZlcnJlZF91c2VybmFtZSI6IjlkZTBjZjJlYmM3ZjgzNDkiLCJlbWFpbCI6IlJvaGl0X1lhZGF2QG1ja2luc2V5LmNvbSIsImFjciI6IjEiLCJzaWQiOiI1MTg3NDNkYi1hY2RiLTQwMGQtYmE1ZC1hYjEyMGFjODg4NzMiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZm1ubyI6IjMyODk5OCIsImdyb3VwcyI6WyJBbGwgRmlybSBVc2VycyIsIjU1MDQ4Yzk4LWUzOTAtNGI2My04ODJmLWUyY2IwZTE4ZmM5ZSJdfQ.gaTGY7DZ7Fmtp_I6gEsYaK2IWGhb2j3k7b-gXlvYMnNUrT3vj-GduQIQpk_E2Q7QU_Kw7FrJeaRvyHv7IhDo5qi8lXSkQJ5N4zR7zQjO43godPTut5GhMLpaRftjsboYCaJ4ASxGrqumv6Sbk3SWChjmsqfCCfICpN4zJ3603aeP8mFEwjOjGsL-pjsvTc--wfzDJocOwou8Xlv3PaDpw04H8cPMAC3pdyqaEtTvKpn1q2KF97TB7N6NJCGY0xOpanO0x0HaMfE5koheYKXRlilWCS4UTy2caE0MrWR6bQvgQRe9HqtsEABkpaDp7f0LXyEqVZQif198OPeGXHM2RQ"
os.environ['OPENAI_BASE_URL'] = "https://openai.prod.ai-gateway.quantumblack.com/55048c98-e390-4b63-882f-e2cb0e18fc9e/v1"

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