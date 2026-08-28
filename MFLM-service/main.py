from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from MFLM.serve_test import load_model, predict
from typing import Annotated
import os
import io
import PIL as Image

app = FastAPI()

def save_mask(mask, MFLM_output_path, img_path):
    os.makedirs(MFLM_output_path, exist_ok=True)
    filename = os.path.basename(img_path)
    save_path = os.path.join(MFLM_output_path, filename)
    mask.save(save_path, format="PNG")

class MFLMRequest(BaseModel):
    image_path: str
    text_output: str = "the image has not been tampered with."
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
        res = predict(req.model_dump())
        save_mask(res["mask"], req["MFLM_output_path"], req["image_path"])
        return res
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mflm/remote/predict", status_code=200)
def handle_remote_mdlm_req(text_output: Annotated[str, Form(...)] = "", img: UploadFile = File(...)):
    try:
        req = {
            "image_path": f"/tmp/{img.filename}",
            "text_output": text_output,
            "MFLM_output_path": "./playground/MFLM_output"
        }

        with open(req["image_path"], "wb") as f:
            f.write(img.file.read())

        res = predict(req)
        buff = io.BytesIO()
        res["mask"].save(buff, format="PNG")
        buff.seek(0)

        headers = {
            "X-pred_label": str(res["pred_label"]),
        }

        # debug:
        save_mask(res["mask"], req["MFLM_output_path"], req["image_path"])

        return StreamingResponse(buff, media_type="image/png", headers=headers)

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))