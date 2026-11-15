import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

# we just put # traceable over every method we want traced 


@traceable(name="basic_trace")
def demo_basic_tracing():

    llm = ChatAnthropic(model="claude-haiku-4-5")

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one scentence."
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic" : "machine learning" })


    print(result)