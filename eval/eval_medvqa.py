import os
import re
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image, ImageDraw
import base64
import io
from openai import OpenAI
import requests
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--image_root', type=str, default='/home/yuexi/projects/Med-VLM-R1/data/', help='Root directory for images')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--eval_jsonl', type=str, default=None, help='Path to the evaluation JSONL file')

args = parser.parse_args()

openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:8004/v1'
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


model_name = args.model_name
save_path = os.path.join(args.save_path, model_name)
eval_model_name = "deepeyes"
os.makedirs(save_path, exist_ok=True)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""
USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_before = """Question: {question}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def process(data_arg):
    image_path, question, gt = data_arg
    prompt = instruction_prompt_before.format(question=question)
    pil_img = Image.open(image_path).convert('RGB')
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_message = messages

    response_message = ""

    status = 'success'
    try_count = 0
    turn_idx = 0
    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if try_count > 10:
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 10240,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list['arguments']['bbox_2d']
                bbox = bbox_str
                left, top, right, bottom = bbox
                cropped_image = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                bbox_list.append(bbox)
                cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    }
                ]

                chat_message.extend(_message)
            
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_msg},
                        ],
                    }
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)
            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message
    # print(output_text, "|", gt)
    
    save_info = {}
    save_info['image'] = image_path
    save_info['question'] = question
    save_info['answer'] = gt
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['status'] = status
    return save_info


if __name__ == "__main__":
    data = []
    with open(args.eval_jsonl) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"### Loaded {len(data)} records")
    
    data_name = os.path.basename(args.eval_jsonl).split('.jsonl')[0]
    
    save_json = []
    
    save_name = f"deepeyes_eval_results_{data_name}_{args.model_name}.jsonl"
    pool = multiprocessing.Pool(processes=args.num_workers)
    data_arg = []
    for item in data:
        if isinstance(item['image'], list):
            image_path = os.path.join(args.image_root, item['image'][0])
        else:
            image_path = os.path.join(args.image_root, item['image'])
        question = item['conversations'][0]['value'].replace('<image>', '').strip()
        gt = item['conversations'][1]['value'].strip()
        data_arg.append((image_path, question, gt))
    
    with tqdm(total=len(data_arg), desc=f"Processing {data_name}") as pbar:
        for result in pool.imap(process, data_arg):
            if result is not None:
                save_json.append(result)
                pbar.update(1)

    pool.close()
    pool.join()
    
    with open(os.path.join(save_path, save_name), 'w') as f:
        for item in save_json:
            f.write(json.dumps(item) + '\n')