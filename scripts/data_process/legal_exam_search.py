"""
Generate parquet from local JSONL dataset with NQ-style schema
"""
import os
import json
import argparse
from datasets import Dataset

def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def make_prefix(dp, template_type):
    q = dp["question"].strip()
    base_instruction = (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between <information> and </information>. "
        "You can search as many times as you want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
        "without detailed illustrations. For example, <answer> Beijing </answer>. "
    )
    base_instruction="""
    根据要求，回答问题。
每次获得新信息后，你必须首先在 <think> 和 </think> 标签内进行推理。
推理完成后，如果你发现自己缺少某些知识，可以通过 <search> 【查询词】 </search> 调用搜索引擎，系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。
你可以根据需要进行多次搜索。
如果你认为无需进一步获取外部知识，即可直接在 <answer> 和 </answer> 标签内提供答案，无需详细说明。例如：<answer> 北京 </answer>。
问题：{question}
"""
    if template_type in ("base", "choice"):
        return f"{base_instruction}Question: {q}\n"
    else:
        raise NotImplementedError(f"Unknown template_type: {template_type}")

def build_dataset(examples, split, template_type, data_source):
    """Build dataset matching original NQ schema"""
    data = []
    for idx, ex in enumerate(examples):
        prompt = [{"role": "user", "content": make_prefix(ex, template_type)}]
        reward_model = {"style": "rule", "ground_truth": {"target": ex["golden_answers"]}}
        extra_info = {"index": idx, "split": split}

        record = {
            "id": ex.get("id"),
            "question": ex["question"],
            "golden_answers": ex["golden_answers"],
            "data_source": data_source,
            "prompt": prompt,
            "ability": "fact-reasoning",
            "reward_model": reward_model,
            "extra_info": extra_info,
        }
        data.append(record)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", default="./output", help="Directory to save parquet")
    parser.add_argument("--data_name", default="train", help="Output parquet filename prefix")
    parser.add_argument("--template_type", default="base")
    parser.add_argument("--data_source", default="legal_exam")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    examples = load_jsonl(args.data_path)
    records = build_dataset(examples, split=args.data_name, template_type=args.template_type, data_source=args.data_source)

    ds = Dataset.from_list(records)
    out_path = os.path.join(args.output_dir, f"{args.data_name}.parquet")
    ds.to_parquet(out_path)

    print(f"Saved {len(ds)} rows to {out_path} (Arrow-native, NQ-style schema)")
