from fastapi import FastAPI
from pydantic import BaseModel
from multiprocessing import shared_memory
import os
import requests
import json
import numpy as np
from PIL import Image

IMAGE_PATH_TO_TEST = "./playground/images/test.png"
MFLM_SERVICE = "http://mflm-api:8002/mflm/predict" # docker-compose service name and port
DTE_FDM_SERVICE = "http://dte-fdm-api:8001/dte-fdm/predict"
MFLM_OUTPUT_PATH = "./playground/MFLM_output"
DTE_FDM_OUTPUT_PATH = "./playground/DTE-FDM_output.jsonl"


def ensure_dir_for_file(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

app = FastAPI()

class Item(BaseModel):
    shm_name: str
    img_shape: list[int]
    img_dtype: str
    mask_dtype: str

def mock_dte_fdm_output():
    # Crea i dati per il file mock del DTE-FDM
    # IMPORTANTE: Evitiamo la frase "has not been tampered with" per forzare l'esecuzione dell'MFLM
    mock_dte_data = {
        "image": IMAGE_PATH_TO_TEST,
        "outputs": "The image has been tampered with. There is a spliced object in the foreground with inconsistent illumination and shadow artifacts."
    }

    ensure_dir_for_file(DTE_FDM_OUTPUT_PATH)

    # writhe file .jsonl
    with open(DTE_FDM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(mock_dte_data) + "\n")

    print(f"✅ File mock creato con successo: {DTE_FDM_OUTPUT_PATH}")
    return True # indicate if the image has been tampered with or not


"""
    input batch (N, H, W, C), where N is the batch size.
    Every image in the batch is expected to be in the range [0, 255] saved as a PNG file.
    the returned mask batch will be (N, H, W, 1) in the range [0, 1] as well.
"""
def inference(img: np.ndarray, mask_dtype: str) -> np.ndarray:
    target_dtype = np.dtype(mask_dtype)
    # collect the batch of masks from the MFLM service one by one, since the MFLM service is not designed to handle batches of images.
    mask_batch = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1), dtype=target_dtype)
    labels = []

    ensure_dir_for_file(MFLM_OUTPUT_PATH + "/test.png")
    
    for i, im in enumerate(img):
        Image.fromarray(im.astype(np.uint8)).save(IMAGE_PATH_TO_TEST, compress_level=0, format="PNG")

        labels.append(1 if mock_dte_fdm_output() else 0)

        try:
            response = requests.post(MFLM_SERVICE, json={
                "image_path": IMAGE_PATH_TO_TEST,
                "DTE_FDM_output_path": DTE_FDM_OUTPUT_PATH,
                "MFLM_output_path": MFLM_OUTPUT_PATH
            }, headers = {"Content-Type": "application/json"}, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Failed to get response from MFLM service: {response.text}")
        except Exception as e:
            raise e

        mask = np.array(Image.open(f"{MFLM_OUTPUT_PATH}/test.png").convert("RGB")[:, :, :1], dtype=target_dtype) / 255.0
        mask_batch[i] = mask

    return mask_batch, labels

@app.post("/pred/", status_code=200)
def predict(item: Item):
    inputOutput_shm = None

    try:
        inputOutput_shm = shared_memory.SharedMemory(name=item.shm_name)
    except FileNotFoundError:
        return {"status": "error", "message": "Shared memory buffer not initialized by client yet."}

    # Create a numpy array view of the shared memory buffer
    img_batch = np.ndarray(shape=item.img_shape, dtype=item.img_dtype, buffer=inputOutput_shm.buf)

    try:
        mask_batch, labels = inference(img_batch, item.mask_dtype)
        print(f"Processato image batch and output mask shape {mask_batch.shape}")
    except Exception as e:
        inputOutput_shm.close()
        return {"status": "error", "message": str(e)}

    
    if mask_batch.nbytes > inputOutput_shm.size:
        inputOutput_shm.close()
        return {"status": "error", "message": "Output mask size exceeds shared memory size."}

    # Create a numpy array view of the shared memory buffer for the output mask and copy the output mask to it
    shm_slice_out = np.ndarray(shape=mask_batch.shape, dtype=mask_batch.dtype, buffer=inputOutput_shm.buf)
    shm_slice_out[:] = mask_batch[:]
    
    del shm_slice_out
    inputOutput_shm.close()

    return {
        "shape": list(mask_batch.shape),
        "dtype": str(mask_batch.dtype),
        "labels": labels,
        "status": "success"
    }