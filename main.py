from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (StrOutputParser,JsonOutputParser)
from langchain_core.runnables import RunnableParallel

"""the chain that we are using is
prompt|model |parser  """

# load api.env file
load_dotenv(
    r"C:\Users\Tanul\OneDrive\Desktop\intern\day 15\api.env"
)

api_key=os.getenv("GEMINI_API_KEY")

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key,
    max_output_tokens=3000
)

summary_prompt=ChatPromptTemplate.from_template(""" create a short summary : {text} """)

parser=StrOutputParser()

chain=(summary_prompt|model|StrOutputParser())

title_prompt=ChatPromptTemplate.from_template("""create a good title : {text}""")

title_chain=(title_prompt|model|StrOutputParser())

keyword_prompt=ChatPromptTemplate.from_template("""extract the 5 important keywords :{text}""")

keyword_chain=(keyword_prompt|model|StrOutputParser())

parallel_chain=RunnableParallel(
    {
        "summary":chain,
        "title":title_chain,
        "keywords":keyword_chain
    }
)

result=parallel_chain.invoke(
    {
        "text":
        """
        Artificial intelligence allows computers
        to perform tasks that normally require
        human intelligence.
        AI is used in healthcare,
        cars and finance.
        Machine learnig is changing the world.
        Crypto and the mutual funds have rose by a lot
        """
    }
)


parallel_chain.config_specs
print(" this is the output from the gpt\n")
print(result)


#prompt template -> controls the instructions 
#gemini model ->  does the reasoning 
#parser -> clean/structured output