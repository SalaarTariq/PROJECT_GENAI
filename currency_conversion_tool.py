from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

import os
os.environ["GOOGLE_API_KEY"]= "AIzaSyAlxt6gz0Y4R-eLb8_KG2LzesnsaRszvOE"

#tool create 
@tool
def get_conversion_factor(base_currency: str, target_currency:str) -> float:
    """This function fetches the currency conversion factor between a given base currency and a target currency"""
    url = f"https://v6.exchangerate-api.com/v6/710ad4f63f9f1ae226cfaf9f/pair/{base_currency}/{target_currency}"

    response=requests.get(url)
    return response.json()


get_conversion_factor.invoke({'base_currency':'USD','target_currency':'PKR'})

@tool
def convert(base_currecny_value: int,conversion_rate:float)-> float:
    """given a currency conversion rate this function calculates the target currency value from a given base currency value"""
    
    return base_currecny_value *conversion_rate

print(convert.invoke({'base_currency_value':10,'conversion_rate':289}))
