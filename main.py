from typing import Union 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle 
from fastapi import FastAPI
import numpy as np 
from pydantic import BaseModel
import os
import twilio.rest
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
from bs4 import BeautifulSoup
import pdfkit
import requests 
import json
from mindee import Client, documents
import random
import re


def read_text_file(file_path): # for food recommendation
    data = ""
    file = open(file_path, "r", encoding="utf8")
    for line in file.readlines():
        data += line
    file.close()
    return data

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
os.chdir("..")
chunks = text_splitter.split_text(text=text)

embeddings = CohereEmbeddings(
    cohere_api_key="0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd"
)
VectorStore = FAISS.from_texts(chunks, embedding=embeddings)


def recommend(food_name):
    

    query = f"My last meal was {food_name}. Suggest me some other food option for dinner and some description of that food. Also give me feedback, if my last meal might be bad. The response should be in this format: '1. Your last meal was <food>. 2. <advice on this food>. 3. '<food name>' 4. '<food name>' is a good option to eat now because it provides '<description of the food with nutrition details in bullet points>'"
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
        response = response.strip()
        final = []
        for line in response.split("\n"):
            if line and line[0].isdigit():
                final.append(line[2:].strip())
        return final

app = FastAPI() # FASTAPI

account_sid  ="AC9b1306b9fc75efcdda145e3b27dc8d7c" # TWILIO
auth_token ="30146903908d0513ff246e97b8b9492a"
client = twilio.rest.Client(account_sid, auth_token)

glucose_model = pickle.load(open("./model.sav", 'rb')) # GLUCOSE PREDICTION

mindee_client = Client(api_key="1c283084c67d6da8a2b4e13c0f126209").add_endpoint( # ocr
    account_name="AbhinavKun",
    endpoint_name="medication_presciption",
)


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

    local_image_urls = ["bars/bars.png", "bars/bpm.png", "bars/calories.png", "bars/glucose.png", "bars/burned.png"]
    report_name = "report.pdf"
    # generate plots 
    data = requests.get("https://apollo-web-th7i.onrender.com/api/meta/weekly").text
    data = json.loads(data)
    bpm_data = data[2]["data"]
    carbs_data = [random.randint(80, 150) for i in range(7)]
    calories_data = data[0]["data"]
    burned_data = data[1]["data"]

    def predict_glucose(calories_, carbs_, bpm_):
        return glucose_model.predict(np.array([[calories_, carbs_, bpm_]]))[0]*18

    glucose_data = np.array(list(map(predict_glucose,calories_data, carbs_data, bpm_data)))
    bars_data = np.array([int(np.sum(glucose_data <= 50)), int(np.sum(glucose_data <= 80)), int(np.sum(glucose_data <= 150)), int(np.sum(glucose_data <= 190))])
    for i in range(1, bars_data.size):
        bars_data[i] -= bars_data[i - 1]
    for i in range(bars_data.size):
        bars_data[i] = max(bars_data[i], 1)
    bars_data = ((bars_data/np.sum(bars_data))*100).astype(int)
    plots.generate_bars(bars_data, local_image_urls[0])
    plots.generate_bpm(bpm_data, local_image_urls[1])
    plots.generate_calories(calories_data, local_image_urls[2])
    plots.generate_glucose(glucose_data, local_image_urls[3])
    plots.generate_expended(burned_data, local_image_urls[4])

    # upload images 

    
    def post_image(img_name):
        url='https://apollo-web-th7i.onrender.com/api/image/upload'
        files={'file':open(img_name, "rb")}
        response=requests.post(url,files=files,timeout=10)
        return json.loads(response.text)["img_url"]

    remote_image_urls = list(map(post_image,local_image_urls))


    # update html 

    with open('pdf_template.html', 'r') as file:
        html_content = file.read()

    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    

    # Find all img elements and update their src attributes
    img_elements = soup.find_all('img')[2:]
    for i, img in enumerate(img_elements):
        img['src'] = remote_image_urls[i]
    time_p = soup.find_all("p", {"class": "time_p"})[0]
    time_p.string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save the updated HTML content to a new file
    with open('updated_template.html', 'w') as file:
        file.write(str(soup))
    
    # generate report 

    config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")

    pdfkit.from_file("updated_template.html", report_name, configuration=config)

    url='https://apollo-web-th7i.onrender.com/api/image/pdf/upload'
    files={'file':open(report_name, "rb")}
    response=requests.post(url,files=files,timeout=10)

    return {"url": json.loads(response.text)["img_url"]}


@app.get("/ocr")
def send_ocr(url: Union[str, None] = None):
    result = mindee_client.doc_from_url(url).parse(
    documents.TypeCustomV1, endpoint_name="medication_presciption")
    return_json = {}
    for field_name, field_value in result.document.fields.items():
        return_json[field_name] = f"{field_value}"
    return json.dumps(return_json)
