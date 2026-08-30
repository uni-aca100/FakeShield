from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from serve_test import load_model, predict
from typing import Annotated
import os
import io
import numpy as np
import cv2

DEBUG_FLAG = os.environ.get("DEBUG_FLAG", "False").lower() in ("true", "1", "True", "TRUE")
LOAD_MODEL_ON_STARTUP = os.environ.get("LOAD_MODEL_ON_STARTUP", "True").lower() in ("true", "1", "True", "TRUE", "")


app = FastAPI()

@app.on_event("startup")
def startup_model():
    if LOAD_MODEL_ON_STARTUP:
        print("= = = = = Loading MFLM model... = = = = =")
        load_model({
            "version": "./weight/fakeshield-v1-22b/MFLM",
        })
        print("= = = = = MFLM model initialized successfully. = = = = =")
    print("= = = = = MFLM startup phase completed. = = = = =")


@app.post("/mflm/predict", status_code=200)
def handle_remote_mdlm_req(text_output: Annotated[str, Form(...)] = "", img: UploadFile = File(...)):
    try:
        np_array = np.frombuffer(img.file.read(), np.uint8)
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        req = {
            "text_output": text_output,
            "image": image_np,
        }

        if image_np is None:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")


        if DEBUG_FLAG:
            cv2.imwrite(f"/tmp/{img.filename}_mdlm.png", image_np)

        res = predict(req)
        buff = io.BytesIO()
        # it's convient to respond with a losslessly compressed mask
        res["mask"].save(buff, format="PNG")
        buff.seek(0)

        headers = { "x-pred-label": str(res["pred_label"]) }
        return StreamingResponse(buff, media_type="image/png", headers=headers)

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))