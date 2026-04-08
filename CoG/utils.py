from prompt_list import *
import json
import time
import openai
import re
import requests
import random
import os
import httpx


import pickle
import numpy as np
from sentence_transformers import util, SentenceTransformer


color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

def get_openai_client(api_key=None):
    """Construct an OpenAI client supporting custom gateways via env OPENAI_BASE_URL."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    kwargs = {}
    if key:
        kwargs["api_key"] = key
    if base_url:
        kwargs["base_url"] = base_url

    trust_env = os.environ.get("OPENAI_TRUST_ENV", "0") == "1"
    http_client = None
    if base_url and not trust_env:
        http_client = httpx.Client(trust_env=False, timeout=60.0)
    if http_client is not None:
        kwargs["http_client"] = http_client

    return openai.OpenAI(**kwargs)

def retrieve_top_docs(query, docs, model, width=3):
    """用于 PoG 原有的实体粗筛 (Entity Pruning)"""
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo", print_in=True, print_out=True):
    if print_in:
        print(color_green + prompt + color_end)


    max_retries = 5  
    base_wait_time = 2  

    for attempt in range(max_retries):
        try:

            messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."}]
            message_prompt = {"role": "user", "content": prompt}
            messages.append(message_prompt)


            client = get_openai_client(opeani_api_keys)


            completion = client.chat.completions.create(
                model=engine, 
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=60 
            )


            if completion is None or completion.choices is None or len(completion.choices) == 0 or completion.choices[0].message is None:

                raise ValueError("Received empty response from API")

            result = completion.choices[0].message.content


            if result is None:
                result = ""

            try:
                token_num = {
                    "total": completion.usage.total_tokens,
                    "input": completion.usage.prompt_tokens,
                    "output": completion.usage.completion_tokens
                }
            except:
                token_num = {"total": 0, "input": 0, "output": 0}

            if print_out:
                print(color_yellow + result + color_end)


            return result, token_num

        except Exception as e:
            print(color_red + f"--- [Attempt {attempt + 1}/{max_retries}] API Error: {e} ---" + color_end)
            

            if attempt == max_retries - 1:
                print(color_red + "--- Max retries reached. Returning empty. ---" + color_end)
                return "[]", {"total": 0, "input": 0, "output": 0}
            

            time.sleep(base_wait_time)


def convert_dict_name(ent_rel_ent_dict, entid_name):
    name_dict = {}
    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        if entid_name[topic_e] not in name_dict.keys():
            name_dict[entid_name[topic_e]] = {}

        for h_t, r_e_dict in h_t_dict.items():
            if h_t not in name_dict[entid_name[topic_e]].keys():
                name_dict[entid_name[topic_e]][h_t] = {}
            
            for rela, e_list in r_e_dict.items():
                if rela not in name_dict[entid_name[topic_e]][h_t].keys():
                    name_dict[entid_name[topic_e]][h_t][rela] = []
                for ent in e_list:
                    if entid_name[ent] not in name_dict[entid_name[topic_e]][h_t][rela]:
                        name_dict[entid_name[topic_e]][h_t][rela].append(entid_name[ent])
    return name_dict

    
def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name, file_lock):
    tt = time.time()-start_time
    dict = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "call_num": call_num, "total_token": all_t['total'], "input_token": all_t['input'], "output_token": all_t['output'], "time": tt}
    
    with file_lock:
        with open("PoG_{}.jsonl".format(file_name), "a", encoding='utf-8') as outfile:
            json_str = json.dumps(dict)
            outfile.write(json_str + "\n")


def extract_add_ent(string):
    first_brace_p = string.find('[')
    last_brace_p = string.rfind(']')
    string = string[first_brace_p:last_brace_p+1]
    try:
        new_string = eval(string)
    except:
        s_list = string.split('\', \'')
        if len(s_list) == 1:
            new_string = [s_list[0].strip('[\'').strip('\']')]
        else:
            new_string = [s.strip('[\'').strip('\']') for s in s_list]
    return new_string

def extract_memory(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    return string

def extract_reason_and_anwer(string):

    try:
        first_brace_p = string.find('{')
        last_brace_p = string.rfind('}')
        if first_brace_p != -1 and last_brace_p != -1:
            string = string[first_brace_p:last_brace_p+1]
            
        answer = re.search(r'"Answer":\s*"(.*?)"', string)
        if answer:
            answer = answer.group(1)
        else:

            match = re.search(r'"Answer":\s*(\[[^\]]+\])', string)
            if match:
                answer = match.group(1)
            else:
                answer = "Null"

        reason_match = re.search(r'"R":\s*"(.*?)"', string)
        reason = reason_match.group(1) if reason_match else "No reason found"
        
        sufficient_match = re.search(r'"Sufficient":\s*"(.*?)"', string)
        sufficient = sufficient_match.group(1) if sufficient_match else "No"
        
        # print("Answer:", answer)
        # print("Reason:", reason)
        # print("Sufficient:", sufficient)
        return answer, reason, sufficient
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return "Null", "Parse Error", "No"

def extract_add_and_reason(string):
    try:
        first_brace_p = string.find('{')
        last_brace_p = string.rfind('}')
        string = string[first_brace_p:last_brace_p+1]
        
        flag_match = re.search(r'"Add":\s*"(.*?)"', string)
        flag = flag_match.group(1) if flag_match else "No"
        
        reason_match = re.search(r'"Reason":\s*"(.*?)"', string)
        reason = reason_match.group(1) if reason_match else ""

        print("Add:", flag)
        print("Reason:", reason)
        if 'yes' in flag.lower():
            return True, reason
        else:
            return False, reason
    except:
        return False, ""

def generate_without_explored_paths(question, subquestions, args):
    prompt = cot_prompt + question 
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    return response, token_num

def break_question(question, args): 
    prompt = subobjective_prompt + question
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    first_brace_p = response.find('[')
    last_brace_p = response.rfind(']')
    if first_brace_p != -1 and last_brace_p != -1:
        response = response[first_brace_p:last_brace_p+1]
    return response, token_num

def get_subquestions(q_mem_f_path, question, args):
    sub_questions, token_num = break_question(question, args)
    with open(q_mem_f_path+'/'+'subq', 'w', encoding='utf-8') as f:
        f.write(str(sub_questions))
    return sub_questions, token_num

def if_finish_list(question, lst, depth_ent_rel_ent_dict, entid_name, name_entid, q_mem_f_path, results, cluster_chain_of_entities, args, model, visited_set=None):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    if all(elem == "[FINISH_ID]" for elem in lst):
        new_lst = []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
    

    all_ent_set = set()
    for dep, ent_rel_ent_dict in depth_ent_rel_ent_dict.items():
        for topic_e, h_t_dict in ent_rel_ent_dict.items():
            all_ent_set.add(topic_e)
            for h_t, r_e_dict in h_t_dict.items():
                for rela, e_list in r_e_dict.items():

                    if all(entid_name[item].startswith('m.') or entid_name[item].startswith('g.') for item in e_list) and len(e_list) > 10:
                        e_list = random.sample(e_list, 10)
                        

                    if len(e_list) > 70:
                        #print('··········exceed 70 entities··········')
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        # print('sentence:', topn_entities)
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        all_ent_set |= (set(e_list))
                    else:
                        all_ent_set |= (set(e_list))  


    if visited_set:
        all_ent_set = {e_id for e_id in all_ent_set if e_id not in visited_set}
    
    clean_candidates_set = set()
    id_count = 0  
    
    for e_id in all_ent_set:
        name = entid_name.get(e_id, e_id)
        is_id = str(name).startswith('m.') or str(name).startswith('g.') or str(name).isdigit()
        
        if not is_id:

            clean_candidates_set.add(e_id)
        else:

            if id_count < 5:
                clean_candidates_set.add(e_id)
                id_count += 1


    if not clean_candidates_set and all_ent_set:
        clean_candidates_set = set(list(all_ent_set)[:5])


    all_ent_set = clean_candidates_set


    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    

    prompt = judge_reverse+question+'\nEntities set to be retrieved: ' + str(list(set(sorted([entid_name[ent_i] for ent_i in new_lst])))) +'\nMemory: '+his_mem+'\nKnowledge Triplets:'+chain_prompt

    cur_call_time += 1
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    for kk in token_num.keys():
        cur_token[kk] += token_num[kk]

    flag, reason = extract_add_and_reason(response)

    if flag:

        other_entities = sorted(list(all_ent_set - set(new_lst)))
        other_entities_name = [entid_name[ent_i] for ent_i in other_entities]
        
        print('filter already', [entid_name[ent_i] for ent_i in new_lst], [entid_name[ent_i] for ent_i in all_ent_set], other_entities_name)


        prompt = add_ent_prompt+question+'\nReason: '+reason+'\nCandidate Entities: ' + str(sorted(other_entities_name))+'\nMemory: '+his_mem

        cur_call_time += 1
        response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

        for kk in token_num.keys():
            cur_token[kk] += token_num[kk]

        add_ent_list = extract_add_ent(response)

        add_ent_list = [name_entid[ent_i] for ent_i in add_ent_list if ent_i in other_entities_name]
        add_ent_list = sorted(add_ent_list)
        if add_ent_list:
            print('add reverse ent:', len(add_ent_list), [entid_name[ent_i] for ent_i in add_ent_list])
            return new_lst, add_ent_list, cur_call_time, cur_token
            
    return new_lst, [], cur_call_time, cur_token

    
def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa}.")
        exit(-1)
    return datas, question_string



class SkeletonRetriever:

    def __init__(self, index_path, model_name='msmarco-distilbert-base-tas-b'):
        print(f"Loading Skeleton Index from {index_path}...")
        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
            self.data = data['data']
            self.corpus_emb = data['embeddings']
            

            if os.path.exists(model_name):
                self.model = SentenceTransformer(model_name)
            else:
                self.model = SentenceTransformer('msmarco-distilbert-base-tas-b')
                
            self.is_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load index ({e}). Speculation will be disabled.")
            self.is_loaded = False

    def retrieve(self, query, k=2):
        if not self.is_loaded: return [], []
        

        query_emb = self.model.encode(query)
        
        scores = util.cos_sim(query_emb, self.corpus_emb)[0]
        
        scores_np = scores.cpu().numpy()
        
        top_k_indices = np.argsort(-scores_np)[:k]
        
        return [self.data[i] for i in top_k_indices], scores_np[top_k_indices]



def speculate_skeleton_chain(question, topic_entity_names, retriever, args, model): 

    token_num = {'total': 0, 'input': 0, 'output': 0, 'calls': 0}
    
    if not retriever or not getattr(retriever, 'is_loaded', False):

        return [], token_num, None, "FAILED"
    
    try:

        masked_question = mask_entities(question, topic_entity_names)
        print(f"\033[96mOriginal: {question}\033[0m")
        print(f"\033[96mMasked:   {masked_question}\033[0m")
        

        exemplars, scores = retriever.retrieve(masked_question, k=5) 
        

        if len(scores) > 0 and scores[0] > 0.92:

            print(f"\033[95mHigh similarity ({scores[0]:.4f}) detected! Copying Exemplar Skeleton.\033[0m")
            print(f"   Exemplar Q: {exemplars[0]['q']}")
            print(f"   Borrowed S: {exemplars[0]['s']}")
            skeleton = [s.strip() for s in exemplars[0]['s'].split('->')]
            return skeleton, token_num, exemplars[0]['s'], "DIRECT_COPY"

        if len(scores) > 0:
            print(f"--- Retrieved Exemplars (Top-1 Score: {scores[0]:.4f}) ---")
            for i, ex in enumerate(exemplars):
                print(f"[{i}] Q: {ex['q']} | S: {ex['s']}")
            print(f"---------------------------\n")
        
        if len(exemplars) < 2:

            return [], token_num, None, "FAILED"


        prompt = speculate_prompt.format(
            q1=exemplars[0]['q'], s1=exemplars[0]['s'],
            q2=exemplars[1]['q'], s2=exemplars[1]['s'],
            question=question
        )

        response, t_num = run_llm(
            prompt, 
            temperature=0.0, 
            max_tokens=150, 
            opeani_api_keys=args.opeani_api_keys, 
            engine=args.LLM_type, 
            print_in=False, 
            print_out=False
        )
        
        for k in t_num: 
            if k != 'calls': 
                token_num[k] += t_num.get(k, 0)
        
        token_num['calls'] += t_num.get('calls', 0)


        clean_resp = response.strip().split('\n')[0]
        if "Output:" in clean_resp:
            clean_resp = clean_resp.split("Output:")[-1].strip()
            
        skeleton = [s.strip() for s in clean_resp.split('->')]
        skeleton = [s for s in skeleton if '[' not in s and len(s) > 1]
        
        if len(skeleton) > 0:
            print(f"\033[94m[Speculation] Generated: {skeleton}\033[0m")
            return skeleton, token_num, clean_resp, "GENERATED"
            
    except Exception as e:
        print(f"[Speculation Warning] Failed: {e}")
        
    return [], token_num, None, "FAILED"

def mask_entities(question, entity_names, mask="[ENT]"):

    masked_q = question
    sorted_names = sorted(entity_names, key=len, reverse=True)
    for name in sorted_names:
        if name and name in masked_q:
            masked_q = re.sub(r'\b' + re.escape(name) + r'\b', mask, masked_q, flags=re.IGNORECASE)
    return masked_q