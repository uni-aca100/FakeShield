
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from multiprocessing import shared_memory
import os
import requests
import numpy as np
from PIL import Image
import io
from gateway_service.mock import mock_inference_DTE_FDM, mock_mflm_inference


MFLM_SERVICE = (os.environ.get("MFLM_SERVICE") or "").strip() or "http://mflm-api:8002/mflm/predict" # docker-compose service name and port
DTE_FDM_SERVICE = (os.environ.get("DTE_FDM_SERVICE") or "").strip() or "http://dte-fdm-api:8001/dte_fdm/predict"
# MFLM_OUTPUT_PATH = "./playground/MFLM_output"
DEBUG_FLAG = os.environ.get("DEBUG_FLAG", "False").lower() in ("true", "1", "True", "TRUE")
TIMEOUT_DTE_FDM = int((os.environ.get("TIMEOUT_DTE_FDM") or "").strip() or "150")


app = FastAPI()

class Item(BaseModel):
    shm_name: str
    img_shape: list[int]
    img_dtype: str
    mask_dtype: str


"""Call the MFLM service to get the mask for a single image."""
def mflm_inference(mask_dtype: np.dtype, image_bytes: io.BytesIO, dte_fdm_output: str):
    response = requests.post(MFLM_SERVICE, files={
        "img": ("test.png", image_bytes, "image/png")
    },
    data={ "text_output": dte_fdm_output },
    timeout=120)

    if response.status_code != 200:
        raise Exception(f"Failed to get response from MFLM service: {response.text}")

    x_pred_label = response.headers.get("x-pred-label")

    if not x_pred_label:
        raise Exception("Missing x-pred-label header in response from MFLM service.")
    elif x_pred_label not in ["0", "1"]:
        raise Exception(f"Invalid x-pred-label header value: {x_pred_label}")

    label = int(x_pred_label)
    mask_bytes = io.BytesIO(response.content)

    if mask_bytes.getbuffer().nbytes == 0:
        raise Exception("Received empty mask from MFLM service.")

    mask = np.asarray(Image.open(mask_bytes).convert("RGB"))[:, :, :1].astype(mask_dtype) / 255.0
    return mask, label

def inference_DTE_FDM(image_bytes: io.BytesIO):
    response = requests.post(DTE_FDM_SERVICE, files={
        "file": ("test.png", image_bytes, "image/png")
    }, timeout=TIMEOUT_DTE_FDM)
    if response.status_code != 200:
        raise Exception(f"Failed to get response from DTE-FDM service: {response.text}")

    output = response.json().get("text_output", "")
    if output == "":
        raise Exception("Missing text_output in response from DTE-FDM service.")

    label = 0 if "has not been tampered with" in output else 1
    return output, label



"""
    input batch (N, H, W, C), where N is the batch size.
    Every image in the batch is expected to be in the range [0, 255] saved as a PNG file.
    the returned mask batch will be (N, H, W, 1) in the range [0, 1] as well.
"""
def inference(img: np.ndarray, mask_dtype: np.dtype):
    # collect the batch of masks from the MFLM service one by one, since the MFLM service is not designed to handle batches of images.
    mask_batch = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1), dtype=mask_dtype)
    labels = []

    for i, im in enumerate(img):
        im = im.astype(np.uint8)
        if im.ndim == 3 and im.shape[-1] == 1:
            # Image.fromarray can't handle a (H, W, 1) array; treat it as grayscale and convert to RGB.
            im = np.repeat(im, 3, axis=-1)

        test_image = Image.fromarray(im)
        if DEBUG_FLAG:
            test_image.save(f"/tmp/test_img_{i}.png", compress_level=0, format="PNG")  # debug check

        image_bytes = io.BytesIO()
        test_image.save(image_bytes, format="PNG", compress_level=0)
        image_bytes.seek(0)

        dte_fdm_output, dte_fdm_label = inference_DTE_FDM(image_bytes)

        if dte_fdm_label == 0:
            # If DTE-FDM predicts the image has not been tampered with, we can skip MFLM inference and use a black mask.
            mask = np.zeros((img.shape[1], img.shape[2], 1), dtype=mask_dtype)
            label = 0
        else:
            image_bytes.seek(0)
            mask, label = mflm_inference(mask_dtype, image_bytes, dte_fdm_output)

        mask_batch[i] = mask
        labels.append(label)

        if DEBUG_FLAG:
            Image.fromarray((mask * 255).astype(np.uint8).squeeze(-1)).save(f"/tmp/test_mask_{i}.png", compress_level=0, format="PNG")
            print(f"label for image {i}: {label}")

    return mask_batch, labels

@app.post("/pred/", status_code=200)
def predict(item: Item):
    inputOutput_shm = None

    try:
        inputOutput_shm = shared_memory.SharedMemory(name=item.shm_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Shared memory buffer not initialized by client yet.")

    # Create a numpy array view of the shared memory buffer
    img_batch = np.ndarray(shape=item.img_shape, dtype=item.img_dtype, buffer=inputOutput_shm.buf)
    print(f"Received image batch with shape {img_batch.shape} and dtype {img_batch.dtype}")
    try:
        mask_batch, labels = inference(img_batch, np.dtype(item.mask_dtype))
        print(f"Processato image batch and output mask shape {mask_batch.shape}")
    except Exception as e:
        inputOutput_shm.close()
        raise HTTPException(status_code=500, detail=str(e))

    
    if mask_batch.nbytes > inputOutput_shm.size:
        inputOutput_shm.close()
        raise HTTPException(status_code=500, detail="Output mask size exceeds shared memory size.")

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