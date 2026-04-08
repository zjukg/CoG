import json
import re
import os
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ================= 配置区域 =================
# 请确保这些路径指向您真实的 JSON 文件位置
DATASETS = {
    'cwq': '../data/cwq_train.json',
    'webqsp': '../data/WebQSP_train.json',
    'grailqa': '../data/grailqa_train.json' 
}

OUTPUT_DIR = './index'
# 指向您服务器上已有的模型路径
MODEL_NAME = 'msmarco-distilbert-base-tas-b'
# ===========================================

def extract_skeleton(sparql):
    """
    从 SPARQL 中提取纯关系链 (通用逻辑 - 增强版)
    兼容: ns:relation, <http://.../relation>, 以及 GrailQA 的 :relation
    """
    if not sparql: return None
    
    sparql_str = str(sparql)
    
    # --- 核心修改：增强正则表达式 ---
    # 1. (?:...) 是非捕获组，匹配前缀
    # 2. 前缀包括: 'ns:', 'kb:', 'http://.../', 以及 GrailQA 特有的 ':'
    # 3. ([a-zA-Z0-9_.]+) 捕获具体的关系名
    pattern = r'(?:ns:|kb:|http://rdf\.freebase\.com/ns/|:)([a-zA-Z0-9_.]+)'
    
    rels = re.findall(pattern, sparql_str)
    
    # 过滤掉 m.xxx (实体ID), type.object, common., rdfs, rdf 等通用前缀
    clean_rels = [
        r for r in rels 
        if not r.startswith('m.') 
        and 'type.object' not in r 
        and 'common.' not in r
        and 'rdfs' not in r
        and 'rdf' not in r
        and not r[0].isdigit() # 过滤掉纯数字
    ]
    
    if not clean_rels: return None
    return " -> ".join(clean_rels)

def parse_webqsp_item(item):
    """解析 WebQSP: 问题在 RawQuestion, SPARQL 在 Parses[0]"""
    q = item.get('RawQuestion')
    sparql = None
    if 'Parses' in item and len(item['Parses']) > 0:
        sparql = item['Parses'][0].get('Sparql')
    return q, sparql

def parse_grailqa_item(item):
    """解析 GrailQA: 优先找 sparql_query (您的数据格式)"""
    q = item.get('question')
    
    # 1. 您的截图显示主要在 sparql_query
    sparql = item.get('sparql_query')
    
    # 2. 备选：有些版本在 graph_query -> sparql
    if not sparql and 'graph_query' in item:
        sparql = item['graph_query'].get('sparql')
        
    return q, sparql

def parse_cwq_item(item):
    """解析 CWQ: 标准格式"""
    q = item.get('question')
    sparql = item.get('sparql')
    return q, sparql

def build_index(dataset_name, input_path, output_path):
    print(f"\n🚀 Processing {dataset_name} from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # --- 结构解包逻辑 ---
    data_list = []
    if dataset_name == 'webqsp':
        # WebQSP 是字典 {"Questions": [...]}
        if isinstance(raw_data, dict) and "Questions" in raw_data:
            data_list = raw_data["Questions"]
            print(f"✅ WebQSP structure detected. Loaded {len(data_list)} items.")
        else:
            print("❌ Error: WebQSP format mismatch (expected 'Questions' key).")
            return
    else:
        # CWQ 和 GrailQA 是列表 [...]
        if isinstance(raw_data, list):
            data_list = raw_data
            print(f"✅ List structure detected. Loaded {len(data_list)} items.")
        else:
            print(f"❌ Error: {dataset_name} format mismatch (expected List).")
            return

    # --- 提取骨架 ---
    unique_skeletons = {}
    
    for item in tqdm(data_list, desc="Extracting"):
        q, sparql = None, None
        
        # 路由到不同的解析器
        if dataset_name == 'webqsp':
            q, sparql = parse_webqsp_item(item)
        elif dataset_name == 'grailqa':
            q, sparql = parse_grailqa_item(item)
        elif dataset_name == 'cwq':
            q, sparql = parse_cwq_item(item)
            
        if q and sparql:
            skel = extract_skeleton(sparql)
            if skel:
                # 去重策略：保留更长的问题（信息量更大）
                if skel not in unique_skeletons:
                    unique_skeletons[skel] = q
                elif len(q) > len(unique_skeletons[skel]):
                    unique_skeletons[skel] = q
    
    if len(unique_skeletons) == 0:
        print(f"⚠️ Warning: Extracted 0 skeletons for {dataset_name}. Check regex or field names.")
        return

    final_data = [{'q': q, 's': s} for s, q in unique_skeletons.items()]
    print(f"📊 Extracted {len(final_data)} unique logic skeletons.")

    # --- 编码向量 ---
    print(f"🧮 Encoding vectors with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([d['q'] for d in final_data], show_progress_bar=True)

    # --- 保存索引 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_obj = {'data': final_data, 'embeddings': embeddings}
    with open(output_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"💾 Index saved to {output_path}")

if __name__ == '__main__':
    # 依次处理三个数据集
    for name in ['cwq', 'webqsp', 'grailqa']:
        path = DATASETS.get(name)
        if path:
            build_index(name, path, f'{OUTPUT_DIR}/{name}_index.pkl')
        else:
            print(f"Skipping {name}: Path not configured.")