import os, json, itertools, bisect, gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time
from peft import PeftModel
from serve.cli import generate_stream
from conversation import SeparatorStyle, conv_templates

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"]="1"
MODEL_NAME = 'llama-7b-hf'

def load_model_lora(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LLaMATokenizer.from_pretrained(model_name)
    model = transformers.LLaMAForCausalLM.from_pretrained(
        model_name,
        #device_map=device_map,
        #device_map="auto",
        torch_dtype=torch.float16,
        #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        #load_in_8bit=eight_bit,
        #from_tf=True,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    # ckpt_list = ["checkpoint-3000", "checkpoint-3200", "checkpoint-3400"]
    # for checkpoint in ckpt_list:
    #      print('Merge checkpint: {}'.format(checkpoint))
    model = PeftModel.from_pretrained(model, "../lora_models")
    model = model.merge_and_unload()

    generator = model.generate

load_model_lora("../pretrained/")

# First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
# First_chat = "ChatDoctor: I'm your friend, you can say anything you want."
# print(First_chat)
history = []
# history.append(First_chat)

def go(input):
    global history
    conv = conv_templates["doctor"].copy()

    assert len(history) % 2 == 0, "History must be an even number of messages"

    for i in range(0, len(history), 2):
        conv.append_message(conv.roles[0], history[i])
        conv.append_message(conv.roles[1], history[i + 1])

    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    params = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }

    prev = len(prompt) + 1
    for outputs in generate_stream(tokenizer, model, params, "cuda"):
         yield outputs[prev:].replace("##", "")
         prev = len(outputs)

    # invitation = "ChatDoctor: "
    # human_invitation = "Patient: "

    # # input
    # msg = input(human_invitation)
    # print("")

    # history.append(human_invitation + msg)

    # fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(history) + "\n\n" + invitation
    # #fulltext = "\n\n".join(history) + "\n\n" + invitation
    
    # #print('SENDING==========')
    # #print(fulltext)
    # #print('==========')

    # generated_text = ""
    # gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    # in_tokens = len(gen_in)
    # with torch.no_grad():
    #         generated_ids = generator(
    #             gen_in,
    #             max_new_tokens=200,
    #             use_cache=True,
    #             pad_token_id=tokenizer.eos_token_id,
    #             num_return_sequences=1,
    #             do_sample=True,
    #             repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
    #             temperature=0.5, # default: 1.0
    #             top_k = 50, # default: 50
    #             top_p = 1.0, # default: 1.0
    #             early_stopping=True,
    #         )
    #         generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

    #         text_without_prompt = generated_text[len(fulltext):]

    # response = text_without_prompt

    # response = response.split(human_invitation)[0]

    # response.strip()

    # print(invitation + response)

    # print("")

    # history.append(invitation + response)

# while True:
#     go()
         
if __name__ == "__main__":
    for val in go("who are you?"):
        print(val, end="", flush=True)