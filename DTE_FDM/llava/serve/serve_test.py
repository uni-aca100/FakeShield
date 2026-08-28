import os
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json

import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

# global
tokenizer = None
model = None
image_processor = None
context_len = None
DTG = None
model_name = None
args = None

class DomainTagGenerator:
    def __init__(self, model_path, num_classes=3, device=None):
        """
        Initialize the DomainTagGenerator class.

        parameter:
        - model_path (str): The path to the model weight file.
        - num_classes (int): The number of categories in the category.
        - device (torch.device, optional): Device type, such as 'cpu' or 'cuda'.        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Classification prediction of single images.

        parameter:
        - image_path (str): The path to the image file.

        return:
        - int: predicted category tags.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        return label

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def _init_args(custom_args: dict):
    """ Initialize the arguments for the DTE-FDM model. """
    class DefaultArgs:
        pass

    default_args = DefaultArgs()
    setattr(default_args, "model_path", "facebook/opt-350m")
    setattr(default_args, "model_base", None)
    setattr(default_args, "DTG_path", "ckp/DTG.pth")
    setattr(default_args, "device", "cuda")
    setattr(default_args, "conv_mode", None)
    setattr(default_args, "temperature", 0.2)
    setattr(default_args, "max_new_tokens", 4096)
    setattr(default_args, "load_8bit", False)
    setattr(default_args, "load_4bit", False)
    setattr(default_args, "debug", False)
    
    for key, value in custom_args.items():
        setattr(default_args, key, value)
    return default_args


def DTE_FDM_init(custom_args: dict):
    global tokenizer, model, image_processor, context_len, DTG, model_name, args

    if model is not None:
        return

    print("======== DTE_FDM Model Loading ========")
    args = _init_args(custom_args)
    disable_torch_init()
    model_name = "llava-v1.5-13b"
    DTG = DomainTagGenerator(model_path=args.DTG_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    print("======== DTE_FDM Model Loaded ========")


def DTE_FDM_predict(req: dict):
    global tokenizer, model, image_processor, context_len, DTG, model_name, args

    if model is None:
        raise Exception("Model is not initialized. Please call DTE_FDM_init() first.")

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image_path = req["image_path"]
    image = load_image(image_path)
    label = DTG.predict(image_path)
    print("======== DTE_FDM predict ========")

    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # inp = input(f"{roles[0]}: ")
    inp = "Was this photo taken directly from the camera without any processing? Has it been tampered with by any artificial photo modification techniques such as ps? Please zoom in on any details in the image, paying special attention to the edges of the objects, capturing some unnatural edges and perspective relationships, some incorrect semantics, unnatural lighting and darkness etc."
    if label == 0:
        inp = "This is a picture that is suspected to have been tampered with by AIGC inpainting. " + inp
    elif label == 1:
        inp = "This is a picture that is suspected to have been tampered with by DeepFake. " + inp
    elif label == 2:
        inp = "This is a picture that is suspected to have been tampered with by Photoshop. " + inp

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("======== DTE_FDM Detect Begin ========")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    outputs = outputs.replace("<s>","").replace("</s>","")

    output_path = req["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"image": image_path, "outputs": outputs}, f)

    # Clear GPU cache to free up memory after inference
    torch.cuda.empty_cache()
    
    print("======== The detection result is saved to {} ========".format(output_path))
    return { "DTE_FDM_output_path": output_path, "text_output": outputs }