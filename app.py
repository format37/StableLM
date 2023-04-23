import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import gradio as gr
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Choose model name
model_name = "stabilityai/stablelm-tuned-alpha-7b" #@param ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]

logger.info(f"Using `{model_name}`")

# Select "big model inference" parameters
torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
load_in_8bit = False #@param {type:"boolean"}
device_map = "auto"

logger.info(f"Loading with: `{torch_dtype=}, {load_in_8bit=}, {device_map=}`")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=getattr(torch, torch_dtype),
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    offload_folder="./offload",
)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Sampling args
max_new_tokens = 128 #@param {type:"slider", min:32.0, max:3072.0, step:32}
temperature = 0.7 #@param {type:"slider", min:0.0, max:1.25, step:0.05}
top_k = 0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
top_p = 0.9 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
do_sample = True #@param {type:"boolean"}

def generate_text(system_prompt, user_prompt):
    prompt = f"{system_prompt}{user_prompt}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)

    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )

    completion_tokens = tokens[0][inputs['input_ids'].size(1):]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    return completion

def main():
    system_prompt_input = gr.inputs.Textbox(lines=4, label="Enter a system prompt")
    user_prompt_input = gr.inputs.Textbox(lines=2, label="Enter a user prompt")
    output_textbox = gr.outputs.Textbox(label="Generated text")

    iface = gr.Interface(
        fn=generate_text,
        inputs=[system_prompt_input, user_prompt_input],
        outputs=output_textbox,
        title="Text Generation with StableLM",
        description="Generate text with the StableLM language model. Enter a prompt and get a response.",
    )

    iface.launch()


if __name__ == "__main__":
    main()
