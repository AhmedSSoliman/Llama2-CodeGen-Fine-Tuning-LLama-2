
# LlaMa2-CodeGen
This model is  [**LlaMa2-7b**](https://huggingface.co/meta-llama/Llama-2-7b) which is fine-tuned on the  [**CodeSearchNet dataset**](https://github.com/github/CodeSearchNet) by using the method  [**QLoRA**](https://github.com/artidoro/qlora) with [PEFT](https://github.com/huggingface/peft) library.

# Model Trained on Google Colab Pro Using AutoTrain, PEFT and QLoRA

## [Implementation code](https://colab.research.google.com/drive/18sAFC7msV0gJ24wn5gl41nU0QRynfLqG?usp=sharing) &nbsp;&nbsp; [![Open in Colab][Colab Badge]][RDP Notebook]


# You can load the LlaMa2-CodeGen model on google colab.





### Example 
```py


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

peft_model_id = "AhmedSSoliman/Llama2-CodeGen-PEFT-QLoRA"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, return_dict=True, load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)




def create_prompt(instruction):
  system = "You are using the Llam2-CodeGen model, a coding assistant that will help the user to resolve the following instruction:\n"
  instruction = "### Input: " + instruction
  return system + "\n" + instruction + "\n\n" + "### Response:" + "\n"

def generate(
        instruction,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = create_prompt(instruction)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #input_ids = inputs["input_ids"].to("cuda")
    #attention_mask = inputs["attention_mask"].to("cuda")

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            #input_ids=input_ids,
            #attention_mask=attention_mask,
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )



    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    stop_output = "### Input"
    gen_response = (generated_response.split(stop_output))[0]


    #s = generation_output.sequences[0]
    #output = tokenizer.decode(s, skip_special_tokens=True)
    #stop_output = "### Input"

    #gen_response = (output.split(stop_output))[0]


    #return output.split("### Response:")[1].lstrip("\n")
    return gen_response






instruction = """
 Write a python code for the name Ahmed to be in a reversed order
"""
print(generate(instruction))
```
