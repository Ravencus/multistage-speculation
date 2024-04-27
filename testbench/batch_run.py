import subprocess
import os
import sys
import time


def generate_test_script(list_tuples):
    # input a list of tuples of (model, assistant_model, digest)
    # for each tuple, generate a test script of the form:
    # CUDA_VISIBLE_DEVICES=0 python -m speculation_tb --model-path model --assistant-model-path assistant_model --model-id "speculative-decoding-oracle[model]-draft[assistant_model]" --bench-name spec_bench --temperature 0.0 --dtype float16 --question-begin 475 --log-dir log
    output_list_script = []
    for model, assistant_model, digest in list_tuples:
        script = f"python -m speculation_tb --model-path {model} --assistant-model-path {assistant_model} --model-id \"{digest}\" --bench-name spec_bench --temperature 0.0 --dtype float16 --log-dir log --question-begin 479 --cache-dir /home/hice1/zzheng345/scratch/model_cache"
        output_list_script.append(script)
        
    return output_list_script

if __name__ == "__main__":
    # generate a list of tuples of (model, assistant_model)
    # set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    list_tuples = [("EleutherAI/pythia-1.4b-deduped", "EleutherAI/pythia-160m-deduped", "pythia-1.4b-160m"), 
                   ("EleutherAI/pythia-2.8b-deduped", "EleutherAI/pythia-160m-deduped", "pythia-2.8b-160m"),
                   ("meta-llama/Llama-2-7b-chat-hf", "PY007/TinyLlama-1.1B-Chat-v0.1", "llama-2-7b-1.1b"),
                   ("meta-llama/Llama-2-13b-chat-hf", "PY007/TinyLlama-1.1B-Chat-v0.1", "llama-2-13b-1.1b"),
                   ("meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", "llama-2-13b-7b"),
                   ("lmsys/vicuna-13b-v1.5", "lmsys/vicuna-7b-v1.5", "vicuna-13b-7b"),
                #    ("lmsys/vicuna-33b-v1.3", "lmsys/vicuna-7b-v1.5", "vicuna-33b-7b"),
                   ]
    list_script = generate_test_script(list_tuples)
    # run the scripts
    for script in list_script:
        subprocess.run(script, shell=True)
        time.sleep(10) # sleep for 10 seconds