import io
import numpy as np
import json
from PIL import Image

def mock_inference_DTE_FDM(image_bytes: io.BytesIO):
    # This is a mock function for DTE-FDM inference, always returns a fixed output and label.
    output = "The picture has been tampered with, specifically in the central region slightly towards the bottom half of the image, where a foreign object has been clearly inserted into the foreground"
    label = 1
    return output, label

def mock_mflm_inference(mask_dtype: np.dtype, image: np.ndarray):
    # This is a mock function for testing purposes.
    # It generates a dummy mask and label.
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=mask_dtype)
    label = 0
    return mask, label