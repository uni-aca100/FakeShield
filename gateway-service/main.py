from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torchvision.utils as tv_utils
import torchvision.io as tv_io
from multiprocessing import shared_memory
import requests

IMAGE_PATH_TO_TEST="./playground/image/test.png"
MFLM_SERVICE = "http://mflm-api:8002/mflm/predict" # docker-compose service name and port
DTE_FDM_SERVICE = "http://dte-fdm-api:8001/dte-fdm/predict"
MFLM_OUTPUT_PATH = "./playground/MFLM_output"


app = FastAPI()

class Item(BaseModel):
    shm_name: str
    shape: list[int]
    dtype: str

"""
    return the mask tensor from the MFLM service given an input image tensor.
    The input image tensor is saved to a temporary file and sent to the MFLM service
    via a POST request. The MFLM service will save the output mask to a specified path,
    which is then loaded and returned as a tensor.
"""
def mllm_inference(img: torch.Tensor) -> torch.Tensor:
    tv_utils.save_image(img, IMAGE_PATH_TO_TEST, normalize=True, value_range=(0, 1))

    #TODO mock the DTE-FDM output for now, since the DTE-FDM service is not yet implemented

    try:
        response = requests.post(MFLM_SERVICE, json={
            "image_path": IMAGE_PATH_TO_TEST,
            "DTE_FDM_output_path": "./playground/DTE-FDM_output.jsonl",
            "MFLM_output_path": MFLM_OUTPUT_PATH
        }, headers = {"Content-Type": "application/json"}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to get response from MFLM service: {response.text}")
    except Exception as e:
        raise e

    # load the output image from the MFLM service output path and convert it to a tensor
    output_img_path = f"{MFLM_OUTPUT_PATH}/test.png"
    try:
        # convert to 0-1 range tensor
        output_tensor = tv_io.read_image(output_img_path).float() / 255.0
    except Exception as e:
        raise Exception(f"Failed to load output tensor from {output_img_path}: {str(e)}")

    return output_tensor 

@app.post("/pred/", status_code=200)
def predict(item: Item):
    inputOutput_shm = None

    try:
        inputOutput_shm = shared_memory.SharedMemory(name=item.shm_name)
    except FileNotFoundError:
        return {"status": "error", "message": "Shared memory buffer not initialized by client yet."}

    dtype = eval(item.dtype)

    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = 1
    for dim in item.shape:
        num_elements *= dim
    required_bytes = num_elements * element_size

    buffer_slice_in = inputOutput_shm.buf[:required_bytes]

    in_tensor = torch.frombuffer(
        buffer_slice_in,
        dtype=dtype
    ).reshape(item.shape)

    buffer_slice_in.release()

    try:
        mask = mllm_inference(in_tensor)
        print(f"Processato tensor con shape {in_tensor.shape} and output mask shape {mask.shape}")
    except Exception as e:
        return {"status": "error", "message": str(e)}

    output_required_bytes = mask.numel() * mask.element_size()
    if output_required_bytes > inputOutput_shm.size:
        return {"status": "error", "message": "Output mask size exceeds shared memory size."}

    buffer_slice_out = inputOutput_shm.buf[:output_required_bytes]

    # Copia la maschera nella shared memory
    mask_tensor_view = torch.frombuffer(
        buffer_slice_out,
        dtype=mask.dtype
    ).reshape(mask.shape)
    mask_tensor_view.copy_(mask)

    # close the shared memory segment after use
    del mask_tensor_view 
    buffer_slice_out.release()
    inputOutput_shm.close()

    # Il server risponde solo dopo aver terminato l'uso del tensor.
    # Questo garantisce che il client non sovrascriva la memoria mentre il server sta leggendo.
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "status": "success"
    }