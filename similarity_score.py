
import argparse
import json
import os
import time
import re
import ast

from tqdm import tqdm
import matplotlib.pyplot as plt
from openai import OpenAI
import httpx
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/query_hard_EN.jsonl")
    parser.add_argument("--model_id", type=str, default="deepseek-r1-distill-llama-70b")

    args = parser.parse_args()
    args.savedir = "./eval_score/gpt-4.1"
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    openai_url = os.getenv("OPENAI_BASE_URL", '')
    api_key = os.getenv("OPENAI_API_KEY", '')
    client = OpenAI(base_url=args.model_url, api_key = args.api_key, http_client=httpx.Client(verify=False), timeout=3600)

    prompt = "You are an impartial judge. Please evaluate which two of the three provided images are most similar in style only. Begin your evaluation by comparing the three images and provide a short explanation. Avoid any position biases and ensure that the order in which the images were presented does not influence your decision. After providing your explanation, output your final verdict by strictly following this format: '[[A]]' if the first and second images are more similar, '[[B]]' if if the first and third images are more similar, '[[C]]' if the second and third images are more similar, and '[[D]]' if all three images have different styles."

    objs = []
    with open(args.data_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            objs.append(data["object"])

    eval_results = []
    A = 0
    B = 0
    C = 0
    D = 0
    for obj in tqdm(objs):
        img1_path = f"gpt-4.1-mini/hard/{obj}/{obj}.jpg"
        img2_path = f"qwen2.5-72b-instruct/hard/{obj}/{obj}.jpg"
        img3_path = f"qwen2.5-32b-instruct/hard/{obj}/{obj}.jpg"
        if not os.path.exists(img1_path) or not os.path.exists(img2_path) or not os.path.exists(img3_path):
            eval_result = {obj: "Miss"}
            eval_results.append(eval_result)
            continue

        image1 = encode_image(img1_path)
        image2 = encode_image(img2_path)
        image3 = encode_image(img3_path)

        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                # "image_url": {"url": image_path},
                                "image_url": {"url": f"data:image/jpg;base64,{image1}"},
                            },
                            {
                                "type": "image_url",
                                # "image_url": {"url": image_path},
                                "image_url": {"url": f"data:image/jpg;base64,{image2}"},
                            },
                            {
                                "type": "image_url",
                                # "image_url": {"url": image_path},
                                "image_url": {"url": f"data:image/jpg;base64,{image3}"},
                            },
                        ],
                    }
                ],
                temperature=0.0
            )

            text = response.choices[0].message.content
            answer = text.split("[[")[-1].split("]]")[0].strip()
            if answer == 'A':
                A += 1
            elif answer == 'B':
                B += 1
            elif answer == 'C':
                C += 1
            elif answer == 'D':
                D += 1
            else:
                answer = 'format error'
        except:
            print("score error: ", obj)
            answer = "error"
        eval_result = {obj: answer}
        eval_results.append(eval_result)
    
    with open(f"{args.savedir}/similarity.jsonl", "w") as f:
        for eval_result in eval_results:
            f.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
        f.write("\n" + json.dumps({"A": A, "B": B, "C": C, "D": D}) + "\n")
