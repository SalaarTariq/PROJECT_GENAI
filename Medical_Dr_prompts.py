from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
template = """
You are an helpfull medical assistant. the user will give you how he is feeling and you will assist him, guide him and tell him to take medicines if needed, you will be like his medical advisor 
You dont have to go very detailed go for maximum 5 lines so that the user dosnt gets bored
{sentence}
"""

prompt = PromptTemplate(
    input_variables=["sentence"],
    template=template,
)

user_text = input("Tell me how are you feeling? ")

result = (prompt | model).invoke({"sentence": user_text})
print(result.content)
