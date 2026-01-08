from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langserve import add_routes
import os


load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

generic_template="Translate the following into {language}:"

prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)

parser=StrOutputParser()

chain =prompt | model |parser

##app defination

app=FastAPI(title="langchain server",
            version="1.0",
            description="Simple server using lanchain runnable interfaces ")


add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8080)