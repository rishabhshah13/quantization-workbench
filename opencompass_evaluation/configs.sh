python run.py --datasets mmlu_ppl \
--hf-path "lesliehd/Mistral-7B-Instruct-v0.2-16-bit-quantized" \
--tokenizer-path "mistralai/Mistral-7B-Instruct-v0.2" \
--model-kwargs device_map='auto' \
--max-seq-len 4096 \
--max-out-len 100 \
--batch-size 2  \
--max-partition-size 40000 \
--max-num-workers 32 \
--num-gpus 1