from openai import OpenAI
import numpy as np
from utils import clean_caption
import json
import os
import torch
from tqdm.auto import tqdm
# Modify OpenAI's API key and API base to use vLLM's API server.
HOSTIP=os.environ.get("T5_API_HOST",'localhost') # 10.119.109.197
openai_api_key = "EMPTY"
openai_api_base = f"http://{HOSTIP}:8000/v1"
import hashlib
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)


model = "t5-v1_1-xxl"
prompts = [
    "Hello my name is",
    "The best thing about vLLM is that it supports many different models"
]

def fetch_embedding(prompt):
        cache_name = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        prompt = clean_caption(prompt)
        prompt = clean_caption(prompt)
        response = client.embeddings.create(input=prompt, model=model)
        
        embedding = np.array(response.data[0].embedding).reshape(-1, 4096)
        return (cache_name, embedding)
def batchrun(local_prompts):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交所有任务
        future_to_prompt = {executor.submit(fetch_embedding, prompt): prompt 
                        for prompt in local_prompts}
        
        # 使用tqdm跟踪进度
        results = []
        for future in tqdm(as_completed(future_to_prompt), 
                        total=len(local_prompts),
                        desc="Fetching embeddings"):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Prompt failed: {prompt[:50]}... Error: {e}")
    return results
if __name__ == "__main__":

    jsonpath = "/home/niubility2/tianning/sampled_videos/sandai/sbench/part_9001/i2v_sample.alpha/part_9001.alpha.jsonl"
    with open(jsonpath,'r') as f:
        local_prompts = [json.loads(line)["motion_prompt"] for line in f]
    ROOT="/home/niubility2/tianning/prompt_t5_cache.l40s.nopad/"
    for prompt in tqdm(local_prompts):
        name, embedding = fetch_embedding(prompt)
        real_embedding  = os.path.join(ROOT, name+'.pth')
        real_embedding, real_embedding_mask = torch.load(real_embedding, map_location='cpu')
        # print(real_embedding.shape, real_embedding_mask.shape)
        target_embedding = real_embedding[real_embedding_mask.bool()]    
        # print(target_embedding.shape)
        # print(name, embedding.shape)
        embedding = torch.from_numpy(embedding).float()
        diff = torch.dist(target_embedding, embedding)
        print(diff)
        assert diff == 0, "should be zero at L40S"

# import torch, os
# new_cache_t5_root= "/home/niubility2/tianning/prompt_t5_cache.l40s.nopad/"
# for name, embedding in results:
#     # we will pad the embedding to 800
#     L = len(embedding)
#     embedding = torch.from_numpy(embedding).float()[None] #(1, L, 4096)
#     mask      = torch.ones((1,L))  #(1, L)
#     embedding = torch.nn.functional.pad(embedding, (0, 0, 0, 800 - L))
#     mask      = torch.nn.functional.pad(mask, (0, 800 - L))
#     path = os.path.join(new_cache_t5_root, name+'.pth')
#     torch.save((embedding, mask), path)