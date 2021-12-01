# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:31:28 2021

@author: mitran
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import  Response



import numpy as np
import cv2
import json
from pydantic import BaseModel

import librosa
from ocr import Ourocr
from abstract_summariser import Abstract_Summ
from extract_summarizer import extractive_summarise

from text2speech import synthesiser


from fastapi.middleware.cors import CORSMiddleware

from starlette.responses import StreamingResponse
from io import BytesIO

from ocrsort import ocrsort





    


def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return frame




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_ocr = Ourocr('./ocrweights/')




@app.get('/hello/')
async def start():
    print("request came")
    return "hello digest"


   


@app.post("/ocr")
async def predict(file: UploadFile = File(...) ):
    
    
    
    
    
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\'  not  image.')    

    contents = await file.read()    
    image = load_image_into_numpy_array(contents)
    
    ocr_output = image_ocr(image,(2.0,1.2))
    print("ocr sorting")
    ocr_output=ocrsort(ocr_output)
    
    sentence=""
    for line in ocr_output:
        for word in line:
            sentence+=word[0]+" "
            
    

    return {
        "filename": file.filename, 
        "sentence": sentence ,
        "output": ocr_output
    }

tts = synthesiser('./TTSweights/')

class TTSData(BaseModel):
    text:str    

@app.post('/podcast')
async def podcast(data:TTSData):
    x=tts(data.text)
    from scipy.io import wavfile

    output = BytesIO()
    wavfile.write(output, 22050, x.astype(np.float32))
    
    
    
    #return Response(content=output, media_type='audio/x-wav')
    
    temp= StreamingResponse(output, media_type='audio/x-wav')
    return temp


Summarizer = Abstract_Summ('./transformerweights/')

class SummarizeData(BaseModel):
    text:str
    option :int
    length :int

@app.post('/summarize')
async def summarize(data:SummarizeData):
    
    if(data.option==0):
        return Summarizer.summarize(data.text)
    else:
        print("extractive")
        summ= extractive_summarise(data.text,data.length)
        
        sentence=""
        for line in summ:
            sentence+=line[-1]+" "
        return sentence
        
   


@app.post('/testing/')
async def test(data:str):
    return data
   



