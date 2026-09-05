import json
import torch
from build_uir import build_uir
from torch_geometric.data import Data
from tqdm import tqdm  # 引入 tqdm 库


def process_jsonl_to_binary(jsonl_file, output_file):
    """
    读取jsonl文件，将每一条数据生成 uIR 并加入标签，最后保存为二进制文件
    :param jsonl_file: jsonl 文件路径
    :param output_file: 输出二进制文件路径
    """
    data_list = []

    # 打开 jsonl 文件
    with open(jsonl_file, "r") as f:
        # 使用 tqdm 来显示进度条
        for line in tqdm(f, desc="Processing jsonl data", unit="line"):
            # 解析 jsonl
            data = json.loads(line.strip())
            code1 = data["Code1"]
            lang1 = data["Category1"]
            code2 = data["Code2"]
            lang2 = data["Category2"]

            # 生成 uIR 数据
            uirs1 = build_uir(code1, language=lang1, label_mode="binary")
            uirs2 = build_uir(code2, language=lang2, label_mode="binary")

            # 这里假设正样本的标签为 1，负样本为 0
            label = 1  # 假设是正样本，你可以根据实际需求调整为负样本

            # 假设 uirs1 和 uirs2 是生成的 uIR 列表，代码假设将它们合并为最终的数据
            data1 = torch.tensor([uir["id"] for uir in uirs1], dtype=torch.long)
            data2 = torch.tensor([uir["id"] for uir in uirs2], dtype=torch.long)

            # 数据合并，并添加标签（标签可以是正样本或者负样本）
            data = Data(x1=data1, x2=data2, y=torch.tensor([label], dtype=torch.long))

            # 将数据添加到列表中
            data_list.append(data)

    # 使用 torch 的 DataLoader 将数据保存为二进制文件
    torch.save(data_list, output_file)
    print(f"Data has been successfully saved to {output_file}")


# 使用示例
process_jsonl_to_binary('../generate_dataset/new_data/func_test_results.jsonl',
                        '../generate_dataset/new_data/logic1func0.pt')
