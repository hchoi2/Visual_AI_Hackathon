import vllm
import re
from vllm.assets.image import ImageAsset
import PIL
import pandas as pd

eval_model_names = ['qwen']

def run_qwen2_5_vl(questions: list[str]):
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm = vllm.LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
    )

    placeholder = "<|image_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n") for question in questions
    ]
    stop_token_ids = None
    return llm, prompts, stop_token_ids

MODEL_FNS = {
    "qwen": run_qwen2_5_vl,
}

main_question_prompt = 'Is this fruit healthy or rotten? First, analyze the fruit, and then finish your response with \\boxed{healthy} or \\boxed{rotten}.'

# Read the CSV file
df = pd.read_csv('./docs/fruit_data.csv')

# Get the image paths and labels from the DataFrame
images = df['image'].tolist()
image_labels = df['label'].tolist()
image_labels = [label.lower() for label in image_labels]

# shuffle and take 50 samples
import random
random.seed(42)
perm = list(range(len(images)))
random.shuffle(perm)
images = [images[i] for i in perm]
image_labels = [image_labels[i] for i in perm]
images = images[:50]
image_labels = image_labels[:50]

def extract_answer(result):
    # \boxed{healthy} or \boxed{rotten}
    match = re.search(r'\\boxed\{(.*)\}', result)
    if match:
        answer = match.group(1)
        if 'rotten' in answer.lower():
            return 'rotten'
        elif 'healthy' in answer.lower():
            return 'healthy'
    return None



for eval_model_name in eval_model_names:
    model_fn = MODEL_FNS[eval_model_name]
    questions = [main_question_prompt] * len(images)
    llm, prompts, stop_token_ids = model_fn(questions)
    sampling_params = vllm.SamplingParams(
        temperature=0.0, max_tokens=256, 
    )
    inputs = [{
        "prompt": prompts[i % len(prompts)],
        "multi_modal_data": {
            'image': PIL.Image.open(image_path)
        },
    } for i, image_path in enumerate(images)]
    results = llm.generate(inputs, sampling_params)
    results_texts = [result.outputs[0].text for result in results]
    pred_labels = [extract_answer(result_text) for result_text in results_texts]
    results_is_correct = [extract_answer(result_text) == image_label for result_text, image_label in zip(results_texts, image_labels)]
    print(results_is_correct)

