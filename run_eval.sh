export OPENAI_BASE_URL="https://openai.wokaai.cn/v1/"
export OPENAI_API_KEY="sk-CjbxGrKmsxyseMGUJ4RqiHZK8Z1MhDtuQzHxQRg1YYndQ07D"

python run_eval.py --model_id Qwen2.5-72B-INT8 --eval_rounds 1
