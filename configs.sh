python run.py --datasets mmlu_ppl \
--hf-path "/home/featurize/work/Transformer-Patcher-main/model_cache/llama2-7b" \
--model-kwargs device_map='auto' \
--max-seq-len 4096 \
--max-out-len 100 \
--batch-size 8  \
--max-partition-size 40000 \
--max-num-workers 32 \
--num-gpus 1