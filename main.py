from typing import Union 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle 
from fastapi import FastAPI
import numpy as np 
from pydantic import BaseModel
import os
from twilio.rest import Client
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
import cohere
import langchain
from langchain.chains.question_answering import load_qa_chain
import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import plots
from datetime import datetime, timedelta
import beautifulsoup as bs4



def read_text_file(file_path): # for food recommendation
    data = ""
    file = open(file_path, "r", encoding="utf8")
    for line in file.readlines():
        data += line
    file.close()
    return data

def recommend(food_name):
    path = "./docs"
    text = ""
    os.chdir(path)
    langchain.verbose = False
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"./{file}"
            text += read_text_file(file_path)
        elif file.endswith(".pdf"):
            file_path = f"./{file}"
            pdf = PdfReader(file_path)
            for page in pdf.pages:
                text += page.extractText()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    embeddings = CohereEmbeddings(
        cohere_api_key="0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd"
    )
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    query = f"My last meal was 2 hours ago and I ate {food_name}. Suggest me some other food option for dinner and some description of that food. Also if I ate something that I shouldn't eat then give reasons why it was bad. The response should be in this format: 'Your last meal was <food> <advice on this food>. So, <food> is a good option to eat now because it provides <description of the food>. '"
    # query = "I am hungry and I want to eat something. I am thinking of eating a burger. Is it a good option if i'm a diabetic?"
    # query = "My calorie goal is 1000. I already ate apple today. Give me a balanced diet for my calorie goal, I'm diabetic so keep that in mind? give me numerical data to back up your answer."

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        client = cohere.Client("0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd")
        llm = Cohere(
            client=client, cohere_api_key="0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd"
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response

app = FastAPI() # FASTAPI

account_sid  ="AC9b1306b9fc75efcdda145e3b27dc8d7c" # TWILIO
auth_token ="2f1452378ac9a5bc6614eade59105fb9"
client = Client(account_sid, auth_token)

glucose_model = pickle.load(open("model.sav", 'rb')) # GLUCOSE PREDICTION


@app.get("/")
def read_root():
    return {"response" : "Nikhil is so hot"}

@app.get("/glucose")
def read_item(carbs: Union[float, None] = None, bpm : Union[float, None] = None, calories: Union[float, None] = None):
    if carbs and bpm and calories:
        return {"glucose" : glucose_model.predict(np.array([[calories, carbs, bpm]]))[0]*18}
    return {}


@app.post("/call")
def make_call():
    call = client.calls.create(
                        twiml='<Response><Say>I am calling from Insuliv. Please take your pills and log it in the app.</Say></Response>',
                        to='+919473293740',
                        from_='+12172366046'
                    )

    return call.sid

@app.get("/recommend")
def recommend_food(food: Union[str, None] = None):
    return {"response": recommend(food)}

@app.get("/report")
def generate_report():
    plots.generate_bars([30, 30, 10, 30], "bars/bars.png")
    plots.generate_bpm([10, 20, 30, 40, 50, 60, 70], "bars/bpm.png")
    plots.generate_calories([2000, 3000, 2500, 2250, 1800, 1900, 1500], "bars/calories.png")
    plots.generate_glucose([10, 20, 40, 50, 60, 70, 80], "bars/glucose.png")

    # generate the glucose wise bar graph (also need )

