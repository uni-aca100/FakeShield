from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from DTE_FDM.llava.serve.serve_test_dte_fdm import DTE_FDM_init, DTE_FDM_predict

app = FastAPI()

class Request(BaseModel):
    image_path: str = "./playground/image/Sp_D_CRN_A_ani0043_ani0041_0373.jpg"
    output_path: str = "./playground/DTE-FDM_output.jsonl"

@app.on_event("startup")
def startup_model():
    DTE_FDM_init({
        "model_path": "./weight/fakeshield-v1-22b/DTE-FDM",
        "DTG_path": "./weight/fakeshield-v1-22b/DTG.pth",
    })
    print("= = = = = DTE-FDM model initialized successfully. = = = = =")

@app.post("/dte_fdm/predict", status_code=200)
def handle_dte_fdm_req(req: Request): 
    try:
        return DTE_FDM_predict(req.model_dump())
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))