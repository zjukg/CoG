from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import os
import json
import time
import re
from sentence_transformers import SentenceTransformer
import concurrent.futures
import threading
from functools import partial

from collections import Counter



import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def repeat_unanswer(dataset, datas, question_string, model_name):
    """Filter answered questions for resume capability"""
    answered_set = set()
    file_path = f'PoG_{dataset}_{model_name}.jsonl'
    
    if not os.path.exists(file_path):
        return datas
        
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                if question_string in data:
                    answered_set.add(data[question_string])
            except Exception:
                continue

    new_data = [x for x in datas if x[question_string] not in answered_set]
    print(f"Remaining questions to process: {len(new_data)}")
    return new_data

def get_one_data(datas, question_string, question):
    for data in datas:
        if data[question_string] == question:
            return [data]

def process_question(data, args, question_string, model, file_lock, retriever):

    local_stats = Counter()

    try:
        start_time = time.time()
        call_num = 0
        all_t = {'total': 0, 'input': 0, 'output': 0}
        best_answer_so_far = "null" 
        
        final_json_to_save = None
        final_chain_to_save = []


        global_visited_entities = set()


        question = data[question_string]
        print(f'[Thread-{threading.current_thread().name}] Processing: {question}')
        

        safe_q = re.sub(r'[^\w\-. ]+', '_', question)[:200]
        q_mem_f_path = f'../mem/{args.dataset}/{args.LLM_type}/{safe_q}'
        if not os.path.exists(q_mem_f_path):
            os.makedirs(q_mem_f_path)
        with open(q_mem_f_path+'/mem', 'w', encoding='utf-8') as f:
            pass


        skeleton_chain = []
        global_skeleton_emb = None 
        
        topic_entity = data['topic_entity']
        topic_entity_names = list(topic_entity.values()) 
        
        if retriever:
            try:

                skeleton_chain, skel_tokens, raw_skeleton_str, spec_type = speculate_skeleton_chain(
                    question, topic_entity_names, retriever, args, model 
                )
                
                local_stats[spec_type] += 1
                
                call_num += skel_tokens.get('calls', 0)
                for k in skel_tokens.keys():
                    if k != 'calls': 
                        all_t[k] += skel_tokens.get(k, 0)
                
                if raw_skeleton_str and model:
                    global_skeleton_emb = model.encode(raw_skeleton_str)
                    
            except Exception as e:
                print(f"[Speculation Warning] Failed: {e}")

        call_num += 1
        sub_questions, token_num = get_subquestions(q_mem_f_path, question, args)
        for kk in token_num.keys():
            all_t[kk] += token_num[kk]

        cluster_chain_of_entities = []
        depth_ent_rel_ent_dict = {}
        
        reverse_rec = {'time': 0, 'ent': []}

        entid_name = {}
        name_entid = {}
        for e_id, e_name in topic_entity.items():
            entid_name[e_id] = e_name
            name_entid[e_name] = e_id


        if len(topic_entity) == 0:
            print("No topic entity. Fallback to Refinement.")
            refined_answer = perform_refinement(question, [], args, all_t)
            save_2_jsonl(question, question_string, refined_answer, [], call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type, file_lock=file_lock)
            return local_stats


        for e_id in topic_entity.keys():
            global_visited_entities.add(e_id)


        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_answered = False
        

        for depth in range(1, args.depth+1):
            

            current_pred_rel = None
            if skeleton_chain and (depth - 1) < len(skeleton_chain):
                current_pred_rel = skeleton_chain[depth - 1]

            current_entity_relations_list = []
            layer_guard_budget = 1 


            if len(topic_entity) > 1:
                try:

                    ent_ids = list(topic_entity.keys())

                    ent_names = [str(topic_entity[eid]) for eid in ent_ids]
                    

                    q_emb = model.encode(question)
                    n_embs = model.encode(ent_names)
                    scores = util.cos_sim(q_emb, n_embs)[0].cpu().numpy()
                    

                    sorted_indices = scores.argsort()[::-1]
                    

                    sorted_topic_entity = {ent_ids[i]: topic_entity[ent_ids[i]] for i in sorted_indices}
                    topic_entity = sorted_topic_entity
                    

                except Exception as e:
                    print(f"Sort failed, keep original order: {e}")

            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    call_num += 1 
                    
                    retrieve_relations, token_num, used_count, track_type = relation_search_prune(
                        entity, 
                        sub_questions, 
                        topic_entity[entity], 
                        pre_relations, 
                        pre_heads[i], 
                        question, 
                        args,
                        model=model, 
                        predicted_relation=current_pred_rel,
                        global_skeleton_emb=global_skeleton_emb,
                        speculation_list=skeleton_chain,
                        guard_budget=layer_guard_budget 
                    )
                    

                    local_stats[track_type] += 1
                    local_stats['total_steps'] += 1

                    layer_guard_budget -= used_count
                    call_num += token_num.get('calls', 0) 
                    

                    for kk in token_num.keys():
                        if kk != 'calls':
                            all_t[kk] += token_num.get(kk, 0)
                    current_entity_relations_list.extend(retrieve_relations)
                i+=1
            

            total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = [], [], [], [], []
            ent_rel_ent_dict = {} 
            

            if not current_entity_relations_list:
                print(f"[Thread-{threading.current_thread().name}] No approved relations. Fallback to Refinement.")
                final_json_to_save = perform_refinement(question, cluster_chain_of_entities, args, all_t, best_answer_so_far)
                final_chain_to_save = cluster_chain_of_entities
                flag_answered = True
                break 
            

            for ent_rel in current_entity_relations_list:
                if ent_rel['entity'] not in ent_rel_ent_dict.keys():
                    ent_rel_ent_dict[ent_rel['entity']] = {}

                if ent_rel['head']:
                    head_or_tail = 'head'
                    entity_candidates_id = entity_search(ent_rel['entity'], ent_rel['relation'], True)
                else:
                    head_or_tail = 'tail'
                    entity_candidates_id = entity_search(ent_rel['entity'], ent_rel['relation'], False)
                
                if len(entity_candidates_id) == 0: continue

                entity_candidates, entity_candidates_id = provide_triple(entity_candidates_id, ent_rel['relation'])
                name_entid.update(dict(zip(entity_candidates, entity_candidates_id)))
                entid_name.update(dict(zip(entity_candidates_id, entity_candidates)))

                if head_or_tail not in ent_rel_ent_dict[ent_rel['entity']].keys():
                        ent_rel_ent_dict[ent_rel['entity']][head_or_tail] = {}
                if ent_rel['relation'] not in ent_rel_ent_dict[ent_rel['entity']][head_or_tail].keys():
                    ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']] = []
                
                for retrive_ent in entity_candidates_id:
                    if retrive_ent not in ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']]:
                        ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']].append(retrive_ent)
                
                total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head)
            
            depth_ent_rel_ent_dict[depth] = ent_rel_ent_dict


            if len(total_candidates) == 0:
                print(f"[Thread-{threading.current_thread().name}] Path pruned to empty (Candidates=0). Fallback to Refinement.")
                final_json_to_save = perform_refinement(question, cluster_chain_of_entities, args, all_t, best_answer_so_far)
                final_chain_to_save = cluster_chain_of_entities
                flag_answered = True
                break
            

            flag, chain_of_entities, entities_id, pre_relations, pre_heads, new_ent_rel_ent_dict, cur_call_time, cur_token = entity_condition_prune(question, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, ent_rel_ent_dict, entid_name, name_entid, args, model)
            cluster_chain_of_entities.append(chain_of_entities)

            call_num += cur_call_time
            for kk in cur_token.keys(): all_t[kk] += cur_token[kk]

            if flag:

                call_num += 1
                token_num = update_memory(question, sub_questions, new_ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args)
                for kk in token_num.keys(): all_t[kk] += token_num[kk]

                call_num += 1

                results, answer, sufficient, token_num = reasoning(question, sub_questions, new_ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args)
                for kk in token_num.keys(): all_t[kk] += token_num[kk]


                ans_is_id = str(answer).startswith('m.') or str(answer).startswith("['m.") or str(answer).startswith('["m.')
                ans_is_valid = str(answer).lower() not in ['null', 'none']

                if ans_is_valid:

                    if best_answer_so_far in ['null', 'none', 'Null', None]:
                        best_answer_so_far = answer
                        
                    elif (str(best_answer_so_far).startswith('m.') or str(best_answer_so_far).startswith('[')) and not ans_is_id:
                        best_answer_so_far = answer
                        
                    elif not (str(best_answer_so_far).startswith('m.') or str(best_answer_so_far).startswith('[')) and ans_is_id:
                        pass 
                        
                    else:
                        best_answer_so_far = answer

                if str(answer).lower() == 'null' or str(answer).lower() == 'none'  or str(answer).startswith('m.') or str(answer).startswith('[\"m.') or str(answer).startswith("['m.") or 'yes' not in str(sufficient).lower():
                    stop = False
                else:
                    stop = True
                
                if stop:
                    print(f"[Thread-{threading.current_thread().name}] Stop at depth {depth}. Answer found.")
                    final_json_to_save = results
                    final_chain_to_save = cluster_chain_of_entities
                    flag_answered = True
                    break
                else:

                    add_ent_list = []
                    if reverse_rec['time']<3:
                        

                        entities_id, add_ent_list, cur_call_time, cur_token = if_finish_list(
                            question, entities_id, depth_ent_rel_ent_dict, entid_name, name_entid, 
                            q_mem_f_path, results, cluster_chain_of_entities, args, model,
                            visited_set=global_visited_entities 
                        )

                        call_num += cur_call_time
                        for kk in cur_token.keys(): all_t[kk] += cur_token[kk]
                        

                        add_ent_list = [ent for ent in add_ent_list if ent not in reverse_rec['ent'] and ent not in global_visited_entities]


                        if add_ent_list:
                            reverse_rec['time'] += 1
                            reverse_rec['ent'] += add_ent_list
                            add_ent_list, add_pre_relations, add_pre_heads, new_ent_rel_ent_dict = add_pre_info(add_ent_list, depth_ent_rel_ent_dict, new_ent_rel_ent_dict, entid_name, name_entid, args) 
                            pre_relations += add_pre_relations
                            pre_heads += add_pre_heads
                            entities_id += add_ent_list

                    if not entities_id or depth>5:
                        print(f"[Thread-{threading.current_thread().name}] Forced stop (Max Depth). Fallback to Refinement.")
                        final_json_to_save = perform_refinement(question, cluster_chain_of_entities, args, all_t, best_answer_so_far)
                        final_chain_to_save = cluster_chain_of_entities
                        flag_answered = True
                        break
                    else:
                        topic_entity = {}
                        for entity in entities_id:
                            if if_topic_non_retrieve(entity): continue
                            if entity.startswith("m."):
                                topic_entity[entity] = entid_name[entity]
                        

                        for e_id in topic_entity.keys():
                            global_visited_entities.add(e_id)

            else:
                print(f"[Thread-{threading.current_thread().name}] Path pruned to empty (Entity Pruning). Fallback to Refinement.")
                final_json_to_save = perform_refinement(question, cluster_chain_of_entities, args, all_t, best_answer_so_far)
                final_chain_to_save = cluster_chain_of_entities
                flag_answered = True
                break
        

        
        is_negative_final = False
        

        current_ans_text = ""
        if final_json_to_save:
            try:
                if isinstance(final_json_to_save, str):
                    parsed = json.loads(final_json_to_save)
                    current_ans_text = parsed.get("A", {}).get("Answer", "").lower()
                else:
                    current_ans_text = str(final_json_to_save).lower()
            except:
                current_ans_text = str(final_json_to_save).lower()
        else:
            is_negative_final = True
            
        negative_keywords = [
            "did not", "does not", "has not", "no record", "unknown", 
            "unclear", "no information", "null", "none", "not directed", 
            "no known movie", "insufficient"
        ]
        
        if any(neg in current_ans_text for neg in negative_keywords):
            is_negative_final = True


        if not flag_answered or is_negative_final:
            if is_negative_final:
                print(f"\033[91m[Safety] Answer is Negative ('{current_ans_text[:30]}...'). Force Refinement.\033[0m")
            else:
                print(f"\033[93m[Safety] Loop finished without answer. Force Refinement.\033[0m")
            
            call_num += 1
            final_json_to_save = perform_refinement(question, cluster_chain_of_entities, args, all_t, best_answer_so_far)
            final_chain_to_save = cluster_chain_of_entities


        if final_json_to_save:
            save_2_jsonl(question, question_string, final_json_to_save, final_chain_to_save, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type, file_lock=file_lock)
        else:
            print(f"[Error] No result generated for: {question}")

    except Exception as e:
        print(f"\n[!!!] Error on question: {question} [Thread-{threading.current_thread().name}]")
        import traceback
        traceback.print_exc()


    return local_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int, default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float, default=0.3, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float, default=0.3, help="the temperature in reasoning stage.")
    parser.add_argument("--depth", type=int, default=4, help="choose the search depth of PoG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool, default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str, default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str, default="", help="your openai api keys.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of threads.")

    args = parser.parse_args()
    
    datas, question_string = prepare_dataset(args.dataset)
    datas = repeat_unanswer(args.dataset, datas, question_string, args.LLM_type)

    model_path = 'msmarco-distilbert-base-tas-b'
    if os.path.isdir(model_path):
        print(f"Loading local SBERT from {model_path}...")
        model = SentenceTransformer(model_path)
    else:
        print(f'Local model not found at {model_path}, downloading from Hub...')
        model = SentenceTransformer('msmarco-distilbert-base-tas-b')
    
    index_path = f'./index/{args.dataset}_index.pkl'
    if os.path.exists(index_path):
        print(f"✅ Skeleton Retriever loaded for {args.dataset}")
        retriever = SkeletonRetriever(index_path, model_name=model_path)
    else:
        print(f"⚠️ Warning: Index not found at {index_path}. Running in ZERO-SHOT Speculation mode.")
        retriever = None

    print(f"Start Running Spec-PoG on {args.dataset} dataset with {args.num_workers} workers.")
    file_lock = threading.Lock()

    global_stats = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        process_func = partial(process_question, 
                               args=args,
                               question_string=question_string,
                               model=model,
                               file_lock=file_lock,
                               retriever=retriever)
        
        results = list(tqdm(executor.map(process_func, datas), total=len(datas), desc="Processing"))
        
        for res in results:
            if res: 
                global_stats += res


    print("All finished.")