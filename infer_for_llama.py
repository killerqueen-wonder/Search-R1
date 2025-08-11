# infer_for_llama.py
import transformers
import torch
import re
from datasets import load_dataset
import requests

# 示例问题（不要修改）
question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# Model ID and device setup
model_id = "/linzhihang/huaiwenpang/legal_LLM/Search-R1/Search-R1/verl_checkpoints/nq-search-r1-ppo-llama3.2-3b-em/actor/global_step_1000"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'

curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prepare the message
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you lack knowledge, call a search engine via <search> query </search>. \
It will return results between <information> and </information>. \
You can search multiple times. \
If no more knowledge is needed, provide the answer in <answer> content </answer>, e.g., <answer> Beijing </answer>. \
Question: {question}\n"""

# Initialize tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# Set pad_token to eos_token for Llama-3
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
print(f'[debug] eos_token_id={eos_token_id}, pad_token_id={pad_token_id}')


def get_latest_query(text):
    """提取最后一个 <search>... 中的内容，且必须完整闭合"""
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_final_answer(text):
    """提取 <answer>...</answer> 中的内容"""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def search(query: str):
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
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return ""

    if not results:
        return "No relevant information found."

    def _passages2string(retrieval_result):
        ref = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            lines = content.split("\n", 1)
            title = lines[0] if lines else ""
            text = lines[1] if len(lines) > 1 else ""
            ref += f"Doc {idx+1}(Title: {title}) {text}\n"
        return ref

    return _passages2string(results[0])


# 自定义停止条件：检测是否生成了 </search>
class StopOnSearchEnd(transformers.StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""

    def __call__(self, input_ids, scores, **kwargs):
        new_token = input_ids[0, -1].item()
        try:
            text = self.tokenizer.decode([new_token], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except:
            text = ""
        self.buffer += text

        # 检查是否以 </search> 结尾（允许空格或换行）
        stripped = self.buffer.strip()
        ends = ["</search>", " </search>", "</search>\n", " </search>\n"]
        return any(stripped.endswith(end.strip()) for end in ends)


print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(f'[prompt]: {prompt}')

cnt = 0
full_context = prompt  # 维护完整上下文

while True:
    # 编码输入
    inputs = tokenizer(full_context, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]


    # 检查是否已包含 </search> 但无新内容？防止无限循环
    if cnt >= 4:
        print("[ERROR] Too many search iterations. Breaking loop.")
        break

    # 设置 stopping criteria
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSearchEnd(tokenizer)])

    # 生成（关键：禁用 eos 停止）
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.7,
        eos_token_id=None,  # 禁用 EOS 提前退出
    )

    # 解码新增内容
    generated_tokens = outputs[0][input_ids.shape[1]:]
    new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if not new_text:
        print("[WARNING] No new text generated.")
        break

    # 检查是否输出了最终答案
    final_answer = extract_final_answer(new_text)
    if final_answer is not None:
        print(f"\n✅ Final Answer: <answer> {final_answer} </answer>")
        break

    # 将新内容追加到上下文
    full_context += new_text

    # 提取最新 query
    query = get_latest_query(full_context)

    
    if query:
        print(f"[INFO] Detected search query: {query}")
        search_results = search(query)
        if not search_results.strip():
            search_results = "No relevant information found."

        # 构造搜索返回文本
        search_return_text = curr_search_template.format(
            output_text=new_text,
            search_results=search_results
        )
        full_context += search_return_text
        cnt += 1
        print(f'[search_text in NO.{cnt}]: {search_return_text}')
    else:
        # 无搜索也无答案 → 可能模型结束了
        print(f"\n⚠️ No search or answer generated. Output: {new_text}")
        break

print("\n################# [Reasoning + Searching Finished] ##################")