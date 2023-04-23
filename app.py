import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import gradio as gr
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_name = "stabilityai/stablelm-tuned-alpha-7b"

logger.info(f"Using `{model_name}`")

torch_dtype = "float16"
load_in_8bit = False
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

def generate_text(system_prompt, user_prompt, top_k, top_p, temperature, max_new_tokens, do_sample):
    prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
    
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
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """
    system_prompt_input = gr.inputs.Textbox(lines=4, label="Enter a system prompt", default=system_prompt)

    user_prompt = "Can you write a song about a pirate at sea?"
    user_prompt_input = gr.inputs.Textbox(lines=2, label="Enter a user prompt", default=user_prompt)

    top_k_slider = gr.inputs.Slider(minimum=0.0, maximum=100.0, step=1, default=0.0, label="Top-k")
    top_p_slider = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.05, default=0.9, label="Top-p")
    temperature_slider = gr.inputs.Slider(minimum=0.0, maximum=1.25, step=0.05, default=0.7, label="Temperature")    
    max_new_tokens_slider = gr.inputs.Slider(minimum=32.0, maximum=3072.0, step=32.0, default=64.0, label="Max new tokens")
    do_sample_slider = gr.inputs.Checkbox(label="Do sample", default=True)
    output_textbox = gr.outputs.Textbox(label="Generated text")

    iface = gr.Interface(
        fn=generate_text,
        inputs=[
            system_prompt_input, 
            user_prompt_input,
            top_k_slider, 
            top_p_slider,
            temperature_slider,
            max_new_tokens_slider,
            do_sample_slider
            ],
        outputs=output_textbox,
        title="Text Generation with StableLM",
        description="Generate text with the StableLM language model. Enter a prompt and get a response.",
    )

    iface.launch()

if __name__ == "__main__":
    main()
