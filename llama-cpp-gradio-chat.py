import gradio as gr
import torch
from llama_cpp import Llama

# Path to the Llama model
model_path = "/path/to/llama3-8B-model.gguf"

# Global variables
model = None
default_system_prompt = (
    "You are a helpful, smart, kind, and efficient AI assistant. "
    "You always fulfill the user's requests to the best of your ability. "
    "Your answers must be concise, precise, and accurate."
)
history = []

# Function to trim the message history if the context size exceeds the limit
def trim_history(history, max_tokens):
    tokens_history = sum(len(msg['content'].split()) for msg in history)
    while tokens_history > max_tokens and history:
        history.pop(0)  # Remove the oldest messages
        tokens_history = sum(len(msg['content'].split()) for msg in history)
    return history

# Function to handle conversation with streaming response
def respond_streaming(user_message, chat_history, system_prompt, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p):
    global history
    
    # Add the user's message to the history
    history.append({"role": "user", "content": user_message})
    
    # Trim the history if necessary
    history = trim_history(history, max_tokens)
    
    # Generate the assistant's response
    messages = [{"role": "system", "content": system_prompt}] + history
    
    response = ""
    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        min_p=min_p,
        stream=True  # Enable streaming response
    )

    for chunk in stream:
        content = chunk["choices"][0]["delta"].get("content", "")
        response += content
        yield chat_history + [(user_message, response)], ""  # Update the interface while generating the response
    
    # Add the final response to the history
    history.append({"role": "assistant", "content": response})
    yield chat_history + [(user_message, response)], ""  # Ensure the complete response is shown at the end

# Function to clear the chat and history
def clear_chat():
    global history
    history = []
    return []

def main():
    global model

    # Detect if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Llama model with CUDA support if available
    model = Llama(model_path=model_path, 
                  n_gpu_layers=30 if torch.cuda.is_available() else 0, 
                  n_ctx=8192, 
                  n_batch=521, 
                  verbose=False, 
                  chat_format="llama-3")

    # Create the chat interface with collapsible settings
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(height=600)
                user_message = gr.Textbox(label="Type your question")
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    send_btn = gr.Button("Send")

                # Add settings below the button
                with gr.Accordion("Settings", open=False):
                    system_prompt_input = gr.Textbox(label="System Prompt", value=default_system_prompt, lines=5)
                    temperature_input = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Temperature")
                    max_tokens_input = gr.Slider(-1, 3584, value=-1, step=1, label="Max Tokens (use -1 for unlimited)")
                    top_k_input = gr.Slider(1, 100, value=40, step=1, label="Top K")
                    repeat_penalty_input = gr.Slider(1.0, 2.0, value=1.1, step=0.01, label="Repeat Penalty")
                    min_p_input = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Min P")
                    top_p_input = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top P")     

                # Event to handle the question and generate a response with streaming
                send_btn.click(
                    fn=respond_streaming, 
                    inputs=[user_message, chatbot, system_prompt_input, temperature_input, max_tokens_input, top_k_input, repeat_penalty_input, min_p_input, top_p_input], 
                    outputs=[chatbot, user_message]
                )

                # Event to clear the chat
                clear_btn.click(fn=clear_chat, outputs=chatbot)

                # Allow sending the question with Enter
                user_message.submit(
                    fn=respond_streaming,
                    inputs=[user_message, chatbot, system_prompt_input, temperature_input, max_tokens_input, top_k_input, repeat_penalty_input, min_p_input, top_p_input],
                    outputs=[chatbot, user_message]
                )

            # Add script for autoscroll
            gr.HTML("""
            <script>
                const chatbox = document.querySelector('gradio-app .gradio-chatbot');
                const observer = new MutationObserver((mutations) => {
                    setTimeout(() => {
                        chatbox.scrollTop = chatbox.scrollHeight;
                    }, 100);  // Add a slight delay to ensure the content is rendered
                });
                observer.observe(chatbox, { childList: true });
            </script>
            """)

    # Launch the Gradio application
    demo.launch()

if __name__ == "__main__":
    main()
