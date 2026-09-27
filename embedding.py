from esm.models.esmc import ESMC
from esm.sdk.api import *
import torch
import os
import pickle
import pandas as pd
import re

def setup_model():

    current_dir_weights = "./esmc_600m_2024_12_v0.pth"
    expected_dir_weights = "data/weights/esmc_600m_2024_12_v0.pth"

    if os.path.exists(current_dir_weights) and not os.path.exists(expected_dir_weights):
        print("检测到权重文件在当前目录，创建符号链接...")
        os.makedirs("data/weights", exist_ok=True)
        os.symlink(os.path.abspath(current_dir_weights), expected_dir_weights)
        print("符号链接创建完成")

    os.environ["INFRA_PROVIDER"] = "True"
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    client = ESMC.from_pretrained("esmc_600m", device=device)
    return client, device

def read_data(filepath):

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                data.append({
                    'id': parts[0],
                    'sequence': parts[1],
                    'efficiency': float(parts[2]) if len(parts) > 2 else None
                })
    return pd.DataFrame(data)

def clean_sequence(sequence):

    sequence = sequence.upper()
    valid_chars = "ACDEFGHIKLMNPQRSTVWYX"
    return ''.join(c if c in valid_chars else 'X' for c in sequence)

def get_esm_embedding(client, device, sequence):

    from esm.tokenization import EsmSequenceTokenizer
    tokenizer = EsmSequenceTokenizer()

    sequence = clean_sequence(sequence)

    try:

        token_ids = tokenizer.encode(sequence)
        protein_tensor = ESMProteinTensor(sequence=torch.tensor(token_ids).to(device))

        logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        esm_embedding = logits_output.embeddings
        return esm_embedding
    except Exception as e:
        print(f"序列编码错误: {e}")
        print(f"问题序列: {sequence[:50]}... (长度: {len(sequence)})")
        return None

def process_all_sequences(data_file, output_pkl):

    client, device = setup_model()

    df = read_data(data_file)
    print(f"成功读取 {len(df)} 条序列数据")

    embeddings_dict = {}
    skipped_sequences = 0

    for idx, row in df.iterrows():
        seq_id = row['id']
        sequence = row['sequence']
        efficiency = row['efficiency']

        embedding = get_esm_embedding(client, device, sequence)

        if embedding is not None:

            embeddings_dict[seq_id] = {
                'sequence': sequence,
                'embedding': embedding.cpu(),
                'efficiency': efficiency
            }
            print(f"已处理序列 {idx+1}/{len(df)}: {seq_id}")
        else:
            skipped_sequences += 1
            print(f"跳过序列 {idx+1}/{len(df)}: {seq_id}")

    with open(output_pkl, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"所有嵌入已保存到 {output_pkl}")
    print(f"成功处理 {len(embeddings_dict)} 条序列，跳过 {skipped_sequences} 条序列")
    return embeddings_dict

if __name__ == "__main__":

    input_file = "train_noCR.txt"

    output_file = "apobec_embeddings_noCR.pkl"

    embeddings = process_all_sequences(input_file, output_file)

    if embeddings:
        sample_id = list(embeddings.keys())[0]
        sample_embedding = embeddings[sample_id]['embedding']
        print(f"\n嵌入维度: {sample_embedding.shape}")
