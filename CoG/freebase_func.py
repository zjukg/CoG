from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
import random
import requests
from prompt_list import *
import json
import time
import openai
import re
import numpy as np # 新增
from sentence_transformers import util # 新增
from sentence_transformers import SentenceTransformer

SPARQLPATH = "http://localhost:8890/sparql"  #your own IP and port

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""

sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  BIND(ns:%s AS ?entity)

  {
    # 优先级 1: 标准名 (Name)
    ?entity ns:type.object.name ?tailEntity .
    FILTER(LANG(?tailEntity) = "" || LANGMATCHES(LANG(?tailEntity), "en"))
    BIND(1 AS ?priority)
  }
  UNION
  {
    # 优先级 2: 别名 (Alias) - 解决 Big Fish 问题的关键
    ?entity ns:common.topic.alias ?tailEntity .
    FILTER(LANG(?tailEntity) = "" || LANGMATCHES(LANG(?tailEntity), "en"))
    BIND(2 AS ?priority)
  }
  UNION
  {
    # 优先级 3: 链接 (SameAs) - 只返回 URL 交给 Python 处理
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    BIND(3 AS ?priority)
  }
}
ORDER BY ASC(?priority)
LIMIT 1"""

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"SPARQL Error: {e}")
        return []

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    # 1. 基础过滤：只处理 m.ID 和 http URL
    if not entity_id.startswith('m.') and not entity_id.startswith('http'):
        return entity_id

    final_name = entity_id 

    # 2. 本地 Freebase 查询 (针对 m.ID)
    if entity_id.startswith('m.'):
        try:
            try:
                sparql_query = sparql_id % (entity_id)
            except TypeError:
                sparql_query = sparql_id % (entity_id, entity_id, entity_id)

            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            
            
            results = sparql.query().convert()
            if "results" in results and "bindings" in results["results"]:
                bindings = results["results"]["bindings"]
                if len(bindings) > 0 and 'tailEntity' in bindings[0]:
                    final_name = bindings[0]['tailEntity']['value']
        except Exception:
            pass

    if "wikidata.org/entity/" in final_name:
        try:
            qid = final_name.split("/")[-1]
            # API URL
            url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&props=labels&languages=en&format=json"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json'
            }
            
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if "entities" in data and qid in data["entities"]:
                            entity_node = data["entities"][qid]
                            if "labels" in entity_node and "en" in entity_node["labels"]:
                                return entity_node["labels"]["en"]["value"] 
                        break 
                    elif resp.status_code == 429: # Too Many Requests
                        time.sleep(1) 
                        continue
                    else:
                        break 
                        
                except requests.exceptions.RequestException:

                    time.sleep(0.5)
                    continue
                    
        except Exception as e:
            pass


    return final_name

def select_relations(string, entity_id, head_relations, tail_relations):
    last_brace_l = string.rfind('[')
    last_brace_r = string.rfind(']')
    
    if last_brace_l < last_brace_r:
        string = string[last_brace_l:last_brace_r+1]

    relations=[]
    try:
        rel_list = eval(string.strip())
    except:
        return False, "Eval Error"

    for relation in rel_list:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": True})
        elif relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": False})
    
    if not relations:
        return False, "No relations found"
    return True, relations

def construct_relation_prune_prompt(question, sub_questions, entity_name, total_relations, args):
    return extract_relation_prompt + question + '\nSubobjectives: ' + str(sub_questions) + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations)


def calculate_entropy(scores):

    try:
        exp_scores = np.exp(scores - np.max(scores)) 
        probs = exp_scores / exp_scores.sum()

        entropy = -np.sum(probs * np.log(probs + 1e-9))

        max_entropy = np.log(len(scores)) if len(scores) > 1 else 1.0
        return entropy / max_entropy
    except:
        return 1.0 


def relation_search_prune(entity_id, sub_questions, entity_name, pre_relations, pre_head, question, args, 
                          model=None, predicted_relation=None, 
                          global_skeleton_emb=None, speculation_list=None, 
                          guard_budget=0):
    

    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()
    
    token_num = {'total': 0, 'input': 0, 'output': 0, 'calls': 0}


    if not total_relations:

        return [], token_num, 0, "EMPTY"

    def clean_rel_str(r):
        if not r: return ""
        return r.replace('.', ' ').replace('_', ' ').strip()


    cand_embs = None
    if model and total_relations:
        cleaned_candidates = [clean_rel_str(r) for r in total_relations]
        cand_embs = model.encode(cleaned_candidates)


    if predicted_relation and model and (cand_embs is not None):
        cleaned_pred = clean_rel_str(predicted_relation)
        pred_emb = model.encode(cleaned_pred)
        
        scores = util.cos_sim(pred_emb, cand_embs)[0].cpu().numpy()
        best_idx = scores.argmax()
        max_score = scores[best_idx]
        entropy = calculate_entropy(scores)
        
        if max_score > 0.88 and entropy < 0.4:
            hit_rel = total_relations[best_idx]
            print(f"\033[92m[Fast Track] Entropy={entropy:.2f} | Spec '{predicted_relation}' -> Hit '{hit_rel}'\033[0m")
            is_head = hit_rel in head_relations

            return [{"entity": entity_id, "relation": hit_rel, "head": is_head}], token_num, 0, "FAST_TRACK" 

    
    if model and total_relations and (cand_embs is not None):
        

        q_emb = model.encode(question)
        score_local = util.cos_sim(q_emb, cand_embs)[0].cpu().numpy()
        final_scores = score_local * 0.60

        if predicted_relation:
            if 'pred_emb' not in locals():
                pred_emb = model.encode(clean_rel_str(predicted_relation))
            score_spec = util.cos_sim(pred_emb, cand_embs)[0].cpu().numpy()
            final_scores += score_spec * 0.25
            
        if global_skeleton_emb is not None:
            score_global = util.cos_sim(global_skeleton_emb, cand_embs)[0].cpu().numpy()
            final_scores += score_global * 0.15
    
        if len(final_scores) > 15:
            print(f"\033[93m[Relation Pruning] Too many ({len(final_scores)}). Sorting & Pruning.\033[0m")
            top_k_indices = final_scores.argsort()[::-1][:15]
            total_relations = [total_relations[i] for i in top_k_indices]
        else:
            top_k_indices = final_scores.argsort()[::-1] 
            total_relations = [total_relations[i] for i in top_k_indices]
        

    prompt = construct_relation_prune_prompt(question, sub_questions, entity_name, total_relations, args)
    token_num['calls'] = 1
    result, llm_tokens = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    
    for k in llm_tokens: 
        if k != 'calls':
            token_num[k] += llm_tokens.get(k, 0)
    
    flag, retrieve_relations = select_relations(result, entity_id, head_relations, tail_relations) 

    if not flag:
        retrieve_relations = []

    
    used_budget = 0 

    if speculation_list and len(retrieve_relations) < 2 and guard_budget > 0:
        selected_rel_names = set([item['relation'] for item in retrieve_relations])
        for spec_rel in speculation_list:
            if used_budget >= guard_budget: break 
            if spec_rel in selected_rel_names: continue
            
            is_head = spec_rel in head_relations
            is_tail = spec_rel in tail_relations
            
            if is_head or is_tail:
                direction = True if is_head else False
                if len(retrieve_relations) >= 2: break
                retrieve_relations.append({"entity": entity_id, "relation": spec_rel, "head": direction})
                print(f"\033[93m[Speculation Guard] Forcing keep: {spec_rel} (Global Budget Used)\033[0m")
                used_budget += 1 

    return retrieve_relations, token_num, used_budget, "SLOW_TRACK"

def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)

    entity_ids = replace_entities_prefix(entities)
    return entity_ids



def provide_triple(entity_candidates_id, relation):
    entity_candidates = []
    for entity_id in entity_candidates_id:

        if entity_id.startswith("m.") or entity_id.startswith("http"): 
            entity_candidates.append(id2entity_name_or_type(entity_id))
        else:
            entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id


    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(ent_id_dict.values())
    return entity_candidates, entity_candidates_id

def update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    candidates_relation = [ent_rel['relation']] * len(entity_candidates)
    topic_entities = [ent_rel['entity']] * len(entity_candidates)
    head_num = [ent_rel['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, question_string, subquestions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args, file_lock):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    call_num += 1
    

    if not cluster_chain_of_entities or not cluster_chain_of_entities[0]:
        print("...Knowledge chains are empty. SWITCHING TO CoT FALLBACK.")
        answer, token_num = generate_without_explored_paths(question, subquestions, args)
    else:
        print("...Using existing knowledge to generate answer.")
        answer, token_num = generate_answer(question, subquestions, cluster_chain_of_entities, args)

    for kk in token_num.keys():
        all_t[kk] += token_num[kk]

    save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type, file_lock=file_lock)

def generate_answer(question, subquestions, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question 
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt
    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    return result, token_num

def if_topic_non_retrieve(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_all_digits(lst):
    for s in lst:
        if not s.isdigit():
            return False
    return True

def entity_condition_prune(question, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, ent_rel_ent_dict, entid_name, name_entid, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    new_ent_rel_ent_dict = {}
    no_prune = ['time', 'number', 'date']
    filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                if is_all_digits(e_list) or rela in no_prune or len(e_list) <= 1:
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    select_ent = sorted_e_list
                else:
                    if all(entid_name[item].startswith('m.') or entid_name[item].startswith('g.') for item in e_list) and len(e_list) > 10:
                        e_list = random.sample(e_list, 10)

                    if len(e_list) > 70:
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        # print('sentence:', topn_entities)

                    prompt = prune_entity_prompt + question +'\nTriples: '
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list)

                    print(f"REL: {rela} | ENTS: {sorted_e_list}")

                    cur_call_time += 1
                    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
                    for kk in token_num.keys():
                        cur_token[kk] += token_num[kk]

                    last_brace_l = result.rfind('[')
                    last_brace_r = result.rfind(']')
                    
                    if last_brace_l < last_brace_r:
                        result = result[last_brace_l:last_brace_r+1]
                    
                    try:
                        result = eval(result.strip())
                    except:
                        result = result.strip().strip("[").strip("]").split(', ')
                        result = [x.strip("'") for x in result]

                    select_ent = sorted(result)
                    select_ent = [x for x in select_ent if x in sorted_e_list]
                    print(f"Selected entities after pruning: {select_ent}")


                if len(select_ent) == 0 or all(x == '' for x in select_ent):
                    continue

                if topic_e not in new_ent_rel_ent_dict.keys():
                    new_ent_rel_ent_dict[topic_e] = {}
                if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                    new_ent_rel_ent_dict[topic_e][h_t] = {}
                if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                    new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                
                for ent in select_ent:
                    if ent in sorted_e_list:
                        new_ent_rel_ent_dict[topic_e][h_t][rela].append(name_entid[ent])
                        filter_tops.append(entid_name[topic_e])
                        filter_relations.append(rela)
                        filter_candidates.append(ent)
                        filter_entities_id.append(name_entid[ent])
                        if h_t == 'head':
                            filter_head.append(True)
                        else:
                            filter_head.append(False)


    if len(filter_entities_id) == 0:
        return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token


    cluster_chain_of_entities = [[(filter_tops[i], filter_relations[i], filter_candidates[i]) for i in range(len(filter_candidates))]]
    return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict, cur_call_time, cur_token





def add_pre_info(add_ent_list, depth_ent_rel_ent_dict, new_ent_rel_ent_dict, entid_name, name_entid, args):
    add_entities_id = sorted(add_ent_list)
    add_relations, add_head = [], []
    topic_ent = set()

    for cur_ent in add_entities_id:
        flag = 0
        for depth, ent_rel_ent_dict in depth_ent_rel_ent_dict.items():
            for topic_e, h_t_dict in ent_rel_ent_dict.items():
                for h_t, r_e_dict in h_t_dict.items():
                    for rela, e_list in r_e_dict.items():
                        if cur_ent in e_list:
                            if topic_e not in new_ent_rel_ent_dict.keys():
                                new_ent_rel_ent_dict[topic_e] = {}
                            if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                                new_ent_rel_ent_dict[topic_e][h_t] = {}
                            if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                                new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                            if cur_ent not in new_ent_rel_ent_dict[topic_e][h_t][rela]:
                                new_ent_rel_ent_dict[topic_e][h_t][rela].append(cur_ent)
                            
                            if not flag:
                                add_relations.append(rela)
                                if h_t == 'head':
                                    add_head.append(True)
                                else:
                                    add_head.append(False)
                                flag = 1


        if not flag:
            print('none pre relation')
            print(cur_ent)
            flag = 1
            add_head.append(-1)
            add_relations.append('')
            if cur_ent not in new_ent_rel_ent_dict.keys():
                new_ent_rel_ent_dict[cur_ent] = {}

    return add_entities_id, add_relations, add_head, new_ent_rel_ent_dict

def update_memory(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()
    prompt = update_mem_prompt + question + '\nSubobjectives: '+str(subquestions)+'\nMemory: ' + his_mem

    chain_prompt = ''
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                chain_prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list) + '\n'

    prompt += "\nKnowledge Triplets:\n" + chain_prompt

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    
    mem = extract_memory(response)
    print(mem)
    with open(q_mem_f_path+'/mem', 'w', encoding='utf-8') as f:
        f.write(mem)
    return token_num


def reasoning(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    prompt = answer_depth_prompt + question + '\nMemory: ' + his_mem

    chain_prompt = ''

    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                chain_prompt += entid_name[topic_e] + ', ' + rela + ', ' + str(sorted_e_list) + '\n'

    prompt += "\nKnowledge Triplets:\n" + chain_prompt

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    
    answer, reason, sufficient = extract_reason_and_anwer(response)
    return response, answer, sufficient, token_num

def perform_refinement(question, cluster_chain_of_entities, args, all_t, best_answer_so_far="Null"):

    print(f"\033[93m[Refinement] Graph search incomplete. Translating IDs and Bridging...\033[0m")
    
    try:

        path_str = ""
        if cluster_chain_of_entities:
            for sublist in cluster_chain_of_entities:
                for chain in sublist:

                    if isinstance(chain, (list, tuple)) and len(chain) >= 3:
                        h_raw, r, t_raw = chain[0], chain[1], chain[2]
                        

                        h_show = str(h_raw)
                        if h_show.startswith('m.'):
                            try:

                                name = id2entity_name_or_type(h_raw)
                                if name != h_raw: 
                                    h_show = f"{name} ({h_raw})" 
                            except: pass

                        t_show = str(t_raw)
                        if t_show.startswith('m.'):
                            try:
                                name = id2entity_name_or_type(t_raw)
                                if name != t_raw: 
                                    t_show = f"{name} ({t_raw})"
                            except: pass
                        
                        path_str += f"{h_show} -> {r} -> {t_show}\n"
                    else:

                        path_str += str(chain) + "\n"
        
        if not path_str.strip(): 
            path_str = "No valid paths retrieved."


        final_prompt = f"""
        Question: "{question}"
        The Knowledge Graph search was incomplete. 
        
        [Graph Evidence]:
        {path_str[:3500]} 
        
        Based on the [Graph Evidence] and your INTERNAL KNOWLEDGE, provide the final answer.
        
        Constraints:
        1. If the answer appears in [Graph Evidence] as an ID (e.g., m.01234), Do NOT output Freebase IDs (starting with 'm.' or 'g.'), you MUST translate it to its real Name using your internal knowledge. Never output raw IDs.
        2. Even if the relationship doesn't perfectly match (e.g. "directed" vs "acted"), if there is a strong candidate entity in the path, OUTPUT IT.
        3. Do NOT output refusal phrases such as "did not", "no record", "unknown", "no information", "null", "none", or any similar negative expressions. Do NOT output full sentences explaining why the answer is missing. 
        4. You MUST provide a best-guess answer. Prioritize entities found in the [Graph Evidence], but if the evidence is missing or incomplete, YOU MUST USE YOUR INTERNAL KNOWLEDGE to answer the question directly.
        5. [Format & Logic] Output the final answer string. It could be an Entity Name, a Date, or a Number.Do NOT output JSON. Do NOT explain. Just output the answer string.
        6. Strictly follow any constraints in the question.

        Answer:"""
        


        guess, t_tok = run_llm(final_prompt, 0.0, 50, args.opeani_api_keys, args.LLM_type, False, False)
        for k in t_tok: all_t[k] += t_tok[k]
        

        guess_ans = guess.replace("Answer:", "").strip().strip('"').strip("'").strip(".").split('\n')[0]
        print(f"[Refinement] LLM Output: {guess_ans}")


        final_ans = "Null"
        is_verified = False

        invalid_keywords = ["null", "unknown", "i don't know", "no answer"]
        if not guess_ans or any(k in guess_ans.lower() for k in invalid_keywords):
            if best_answer_so_far not in ["Null", None]:
                final_ans = best_answer_so_far
        else:

            if not (guess_ans.isdigit() or re.match(r'^\d{4}-\d{2}-\d{2}$', guess_ans)):

                 check_sparql = f"""PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?mid WHERE {{ {{ ?entity ns:type.object.name "{guess_ans}"@en . }} UNION {{ ?entity ns:common.topic.alias "{guess_ans}"@en . }} }} LIMIT 1"""
                 try:
                     if execurte_sparql(check_sparql): is_verified = True
                 except: pass


        is_sufficient = "Yes" if (final_ans and "null" not in final_ans.lower()) else "No"
        
        final_result_dict = {
            "A": {"Sufficient": is_sufficient, "Answer": final_ans},
            "R": f"Refined answer with ID decoding: {final_ans}"
        }


        return json.dumps(final_result_dict, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"[Refinement Critical Error] {e}")

        err_dict = {"A": {"Sufficient": "No", "Answer": "Null"}, "R": f"Error: {str(e)}"}
        return json.dumps(err_dict, indent=2, ensure_ascii=False)
    
