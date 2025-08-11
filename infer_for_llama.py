# search_reasoning_agent.py
import transformers
import torch
import random
from datasets import load_dataset
import requests
import re

# 示例问题
question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# Model ID and device setup
# model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
model_id = "/linzhihang/huaiwenpang/legal_LLM/Search-R1/Search-R1/verl_checkpoints/nq-search-r1-ppo-llama3.2-3b-em/actor/global_step_1000"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'

# 搜索结果模板
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prompt 构造
prompt = f"""Answer the given question. \
You must conduct reasoning inside  <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# 设置 pad_token（Llama-3 默认没有 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 当前模型的 eos_token_id（用于最终判断，但 generate 中禁用）
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
curr_eos = [eos_token_id]  # 用于最终生成结束判断（可选）

print(f'[debug] eos_token_id={eos_token_id}, pad_token_id={pad_token_id}')


def get_query(text):
    """从文本中提取最后一次 <search>...</search> 中的查询"""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None


def search(query: str):
    """调用本地搜索引擎 API"""
    if not query or not query.strip():
        print("[WARNING] Empty query passed to search function.")
        return ""

    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8006/retrieve",
            json=payload,
            proxies={"http": None, "https": None},
            timeout=10
        )
        response.raise_for_status()
        json_data = response.json()
        results = json_data.get('result', [])
    except requests.exceptions.Timeout:
        print("[ERROR] Search request timed out.")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return ""
    except ValueError as e:
        print(f"[ERROR] Failed to decode JSON: {e}")
        print("Response content:", response.text[:500])
        return ""

    if not results:
        print("[INFO] No results returned from search.")
        return ""

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            lines = content.split("\n")
            title = lines[0] if lines else ""
            text = "\n".join(lines[1:]) if len(lines) > 1 else ""
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# 自定义停止条件：基于字符串匹配，检测是否生成了 </search>
class StopOnSearchEnd(transformers.StoppingCriteria):
    def __init__(self, tokenizer, target_sequences):
        self.tokenizer = tokenizer
        self.target_sequences = [t.strip() for t in target_sequences]
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        # 解码最新生成的 token
        new_token_id = input_ids[0, -1].item()
        try:
            new_text = self.tokenizer.decode([new_token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except:
            new_text = ""

        self.buffer += new_text

        # 检查 buffer 是否以任意目标序列结尾
        current = self.buffer.strip()
        for target in self.target_sequences:
            if current.endswith(target):
                return True
        return False


# 开始推理循环
print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(f'[prompt]: {prompt}')

cnt = 0
while True:
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 每次生成都新建 stopping_criteria，避免 buffer 累积
    target_endings = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([
        StopOnSearchEnd(tokenizer, target_endings)
    ])

    # 生成：关键点 —— 禁用 eos_token_id 停止机制
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.7,
        eos_token_id=None,  # 不要因生成 EOS 而提前退出
        top_p=0.9,
    )

    # 解码新增内容
    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 获取完整上下文
    full_response = prompt + output_text

    # 提取最新 query
    query = get_query(full_response)

    if query:
        print(f"[INFO] Detected search query: {query}")
        search_results = search(query)
        if not search_results.strip():
            search_results = "No relevant information found."

        # 构造搜索结果文本
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        print(f'[search_text in NO.{cnt}]: {search_text}')
    else:
        # 没有触发搜索，检查是否输出了最终答案
        if "<answer>" in output_text:
            answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                print(f"\n✅ Final Answer: <answer> {final_answer} </answer>")
            else:
                print(f"\n✅ Final Output: {output_text}")
            break
        else:
            # 既无搜索也无答案，可能是模型终止
            print(f"\n⚠️ Generation ended without search or answer:\n{output_text}")
            break

print("\n################# [Reasoning + Searching Finished] ##################")