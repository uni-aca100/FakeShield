from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from MFLM.serve_test import load_model, predict
import os

app = FastAPI()

class MFLMRequest(BaseModel):
    image_path: str
    DTE_FDM_output_path: str = "./playground/DTE-FDM_output.jsonl"
    MFLM_output_path: str = "./playground/MFLM_output"

@app.on_event("startup")
def startup_model():
    load_model_flag = os.environ.get("LOAD_MODEL_ON_STARTUP", "True")
    
    if load_model_flag == "True" or load_model_flag == "":
        print("= = = = = Loading MFLM model... = = = = =")
        load_model({
            "version": "./weight/fakeshield-v1-22b/MFLM",
        })
        print("= = = = = MFLM model initialized successfully. = = = = =")
    print("= = = = = MFLM startup phase completed. = = = = =")

@app.post("/mflm/predict", status_code=200)
def handle_mdlm_req(req: MFLMRequest): 
    try:
        return predict(req.model_dump())
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))