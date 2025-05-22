import pandas as pd
import textwrap
import os

# # 定义数据集到任务的映射
# dataset2task = {
#     'arxiv': ['lp', 'nc'], 'chemblpre': ['gc'], 'chemhiv': ['gc'], 'chempcba': ['gc'],
#     'children': ['lp', 'nc'], 'citeseer': ['nc'], 'computer': ['lp', 'nc'],
#     'cora': ['lp', 'nc'], 'cora_simplified': ['lp', 'nc'], 'fb15k_237': ['lc'],
#     'history': ['lp', 'nc'], 'instagram': ['nc'], 'photo': ['lp', 'nc'], 'products': ['lp', 'nc'],
#     'pubmed': ['lp', 'nc'], 'reddit': ['nc'], 'sports': ['lp', 'nc'], 'wikics': ['nc'], 'wn18rr': ['lc']
# }
#
# train_dataset = ['arxiv', 'citeseer', 'pubmed',
#                  'instagram',
#                  'children', 'computer', 'photo', 'sports',
#                  'chemblpre', 'chempcba',
#                  'wn18rr']
#
# test_dataset = ['cora', 'cora_simplified',
#                 'reddit',
#                 'history', 'products',
#                 'chemhiv',
#                 'fb15k_237',
#                 'wikics']


# 创建输出目录
output_dir = '/home/wuyicong/wyc/graph_reasoning/LLaMA-Factory/data'
os.makedirs(output_dir, exist_ok=True)

# 构造输入文件路径
data_path = f'/home/wuyicong/wyc/graph_reasoning/lgy/data/train_data/sft_train_data_filter2048_10000_balance.csv'

try:
    df = pd.read_csv(data_path)
    print(f'train dataset for sft:{len(df)}')  # 9403
    # sample_dict = {}
    # grouped = df.groupby(['dataset', 'task'])
    # for (dataset, task), group_df in grouped:
    #     if dataset not in sample_dict:
    #         sample_dict[dataset] = {}
    #     sample_answer = group_df['ground_truth'].sample(n=1, random_state=42).iloc[0]
    #     sample_dict[dataset][task] = sample_answer

    output_path = os.path.join(output_dir, f'graph_10000_balance.json')

    df.to_json(output_path, orient='records', lines=True)
    print(f'Successfully processed and saved: {output_path}')

except FileNotFoundError:
    print(f'File not found: {data_path}')
except KeyError:
    print(f'Column "ground_truth" "reasoning" "answer" or "prompt" not found in {data_path}')
except Exception as e:
    print(f'Error processing {data_path}: {e}')
