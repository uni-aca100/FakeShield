from fastapi import FastAPI, HTTPException, File, UploadFile
from DTE_FDM.llava.serve.serve_test import DTE_FDM_init, DTE_FDM_predict
import os
from pathlib import Path
from PIL import Image

app = FastAPI()


@app.on_event("startup")
def startup_model():
    load_model_flag = os.environ.get("LOAD_MODEL_ON_STARTUP", "True")
    
    if load_model_flag == "True" or load_model_flag == "":
        print("= = = = = Loading DTE-FDM model... = = = = =")
        DTE_FDM_init({
            "model_path": "./weight/fakeshield-v1-22b/DTE-FDM",
            "DTG_path": "./weight/fakeshield-v1-22b/DTG.pth",
            "load_8bit": True if os.environ.get("DTE_FDM_LOAD_8BIT", "False") == "True" else False,
            "load_4bit": True if os.environ.get("DTE_FDM_LOAD_4BIT", "False") == "True" else False
        })
        print("= = = = = DTE-FDM model initialized successfully. = = = = =")
    print("= = = = = DTE-FDM startup phase completed. = = = = =")


"""
    input: a png image 
    output json: { "text_output": str }
"""
@app.post("/dte_fdm/predict", status_code=200)
def handle_dte_fdm_remote_req(file: UploadFile = File(...)):
    try:
        file.file.seek(0)
        image = Image.open(file.file).convert('RGB')
        return DTE_FDM_predict(image)
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))