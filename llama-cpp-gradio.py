import gradio as gr
import torch
from llama_cpp import Llama

# Path to the Llama model
model_path = "/path/to/llama3-8B-model.gguf"

# Global variables
default_system_prompt = (
    "You are a helpful, smart, kind, and efficient AI assistant. "
    "You always fulfill the user's requests to the best of your ability. "
    "Your answers must be concise, precise, and accurate."
)
model = None

# Function to generate the model's response
def generate_response(prompt, system_prompt, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p):
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        min_p=min_p,
    )
    return response["choices"][0]["message"]["content"]

# Gradio interface function
def gradio_interface(question, system_prompt, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p):
    # Generate response
    response = generate_response(question, system_prompt, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p)
    # Display question and response in the desired format
    return f"{response}"

def main():
    global model

    # Detect if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Llama model with CUDA support if available
    model = Llama(model_path=model_path, 
                  n_gpu_layers=30 if torch.cuda.is_available() else 0, 
                  n_ctx=3584, 
                  n_batch=521, 
                  verbose=False, 
                  chat_format="llama-3")

    # Gradio Interface
    gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Question"),
            gr.Textbox(label="System Prompt", value=default_system_prompt, lines=5),
            gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Temperature"),
            gr.Slider(-1, 3584, value=-1, step=1, label="Max Tokens (use -1 for unlimited)"),
            gr.Slider(1, 100, value=40, step=1, label="Top K"),
            gr.Slider(1.0, 2.0, value=1.1, step=0.01, label="Repeat Penalty"),
            gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Min P"),
            gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top P"),
        ],
        outputs=gr.Textbox(label="Output"),
        title="LLaMA Response Generator",
        description="Configure the model parameters and get answers to your questions.",
    ).launch()

if __name__ == "__main__":
    main()
