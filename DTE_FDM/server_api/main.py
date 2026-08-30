from fastapi import FastAPI, HTTPException, File, UploadFile
from llava.serve.serve_test import DTE_FDM_init, DTE_FDM_predict
import os
from PIL import Image

LOAD_MODEL_ON_STARTUP = os.environ.get("LOAD_MODEL_ON_STARTUP", "True").lower() in ("true", "1", "True", "TRUE", "")
DTE_FDM_LOAD_8BIT = os.environ.get("DTE_FDM_LOAD_8BIT", "False").lower() in ("true", "1", "True", "TRUE")
DTE_FDM_LOAD_4BIT = os.environ.get("DTE_FDM_LOAD_4BIT", "False").lower() in ("true", "1", "True", "TRUE")
MODEL_PATH = (os.environ.get("MODEL_PATH") or "").strip() or "./weight/fakeshield-v1-22b/DTE-FDM"
DTG_PATH = (os.environ.get("DTG_PATH") or "").strip() or "./weight/fakeshield-v1-22b/DTG.pth"

app = FastAPI()

@app.on_event("startup")
def startup_model():
    
    if LOAD_MODEL_ON_STARTUP:
        print("= = = = = Loading DTE-FDM model... = = = = =")
        DTE_FDM_init({
            "model_path": MODEL_PATH,
            "DTG_path": DTG_PATH,
            "load_8bit": DTE_FDM_LOAD_8BIT,
            "load_4bit": DTE_FDM_LOAD_4BIT
        })
        print("= = = = = DTE-FDM model initialized successfully. = = = = =")
    print("= = = = = DTE-FDM startup phase completed. = = = = =")

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}

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