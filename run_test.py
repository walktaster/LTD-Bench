
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

from evaluation.easy_generation import easy_gen
from evaluation.easy_recognition import easy_rec
from evaluation.normal_generation import normal_gen
from evaluation.normal_recognition import normal_rec
from evaluation.hard_generation import hard_gen




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/question.jsonl")
    parser.add_argument("--model_id", type=str, default="qwq-32b")

    args = parser.parse_args()

    base_url = os.getenv("API_BASE_URL", '')
    api_key = os.getenv("API_KEY", '')
    client = OpenAI(base_url=base_url, api_key = api_key, http_client=httpx.Client(verify=False), timeout=3600)

    easy_gen_questions = []
    easy_rec_questions = []
    normal_gen_questions = []
    normal_rec_questions = []
    hard_gen_questions = []
    with open(args.data_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            if data["level"] == "easy":
                if data["task"] == "generation":
                    easy_gen_questions.append(data)
                elif data["task"] == "recognition":
                    easy_rec_questions.append(data)
                else:
                    continue
            elif data["level"] == "normal":
                if data["task"] == "generation":
                    normal_gen_questions.append(data)
                elif data["task"] == "recognition":
                    normal_rec_questions.append(data)
                else:
                    continue
            elif data["level"] == "hard":
                if data["task"] == "generation":
                    hard_gen_questions.append(data)
                else:
                    continue
            else:
                continue

    print("Starting easy generation...")
    easy_gen(easy_gen_questions, args.model_id, client)
    print("Easy generation done!" + '\n')

    print("Starting easy recognition...")
    easy_rec(easy_rec_questions, args.model_id, client)
    print("Easy recognition done!" + '\n')

    print("Starting normal generation...")
    normal_gen(normal_gen_questions, args.model_id, client)
    print("Normal generation done!" + '\n')

    print("Starting normal recognition...")
    normal_rec(normal_rec_questions, args.model_id, client)
    print("Normal recognition done!" + '\n')

    print("Starting hard generation...")
    hard_gen(hard_gen_questions, args.model_id, client)
    print("Hard generation done!" + '\n')

    print("All done!")
