# CoG
[Paper][ACL 2026] *CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs*

## Knowledge Graph and Datasets

Before running **CoG**, please set up **Freebase** on your local machine by following the [installation guide](https://github.com/GasolSun36/ToG/tree/main/Freebase).

We evaluate **CoG** on **CWQ**, **WebQSP**, and **GrailQA**. The corresponding data files should be placed in the `data/` directory.

## Code

Our codebase is built with reference to the open-source project [PoG](https://github.com/liyichen-cly/PoG). We sincerely appreciate the authors for sharing their implementation.
## Running

After completing all necessary configurations, you can run **CoG** using the following command:

```bash
python main_freebase.py \
  --dataset cwq \ # the dataset
  --max_length 4096 \ # the max length of LLMs output
  --temperature_exploration 0.3 \ # the temperature in exploration stage
  --temperature_reasoning 0.3 \ # the temperature in reasoning stage
  --depth 4 \
  --remove_unnecessary_rel True \ # whether removing unnecessary relations
  --LLM_type gpt-3.5-turbo \ # the LLM
  --opeani_api_keys sk-xxxx \ # your own api keys
  --num_workers 10 
```
## Evaluation

We adopt **Exact Match** as the evaluation metric. After generating the final prediction file, you can evaluate the results with the following example command:

```bash
python eval.py \
  --dataset cwq \
  --output_file CoG_cwq_gpt-3.5-turbo.jsonl
```
