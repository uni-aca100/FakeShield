from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from DTE_FDM.llava.serve.serve_test import DTE_FDM_init, DTE_FDM_predict
import os
from pathlib import Path
import torch

app = FastAPI()

class Request(BaseModel):
    image_path: str = "./playground/images/Sp_D_CRN_A_ani0043_ani0041_0373.jpg"
    output_path: str = "./playground/DTE-FDM_output.jsonl"

@app.on_event("startup")
def startup_model():
    if os.environ.get("LOAD_MODEL_ON_STARTUP", "True") == "True":
        print("= = = = = Loading DTE-FDM model... = = = = =")
        DTE_FDM_init({
            "model_path": "./weight/fakeshield-v1-22b/DTE-FDM",
            "DTG_path": "./weight/fakeshield-v1-22b/DTG.pth",
        })
        print("= = = = = DTE-FDM model initialized successfully. = = = = =")
    print("= = = = = DTE-FDM startup phase completed. = = = = =")

def _dte_fdm_predict(req: Request):
    try:
        return DTE_FDM_predict(req.model_dump())
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dte_fdm/predict", status_code=200)
def handle_dte_fdm_req(req: Request): 
    return _dte_fdm_predict(req)


IMAGE_PATH_TO_TEST = Path("./playground/images/test.png")

"""
    input: a png image 
    output json: { "text_output": str }
"""
@app.post("/dte_fdm/remote/predict", status_code=200)
def handle_dte_fdm_remote_req(file: UploadFile = File(...)):
    try:
        with open(IMAGE_PATH_TO_TEST, "wb") as f:
            f.write(file.file.read())
        req = Request(image_path=str(IMAGE_PATH_TO_TEST))
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return _dte_fdm_predict(req)