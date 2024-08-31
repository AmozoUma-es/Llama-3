import gradio as gr
import faiss
import os
import pickle
import torch
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModel
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Solves OpenMP error

# Path to the Llama model
model_path = "/path/to/llama3-8B-model.gguf"

# Global variables
default_system_prompt = (
    "You are a helpful, smart, kind, and efficient AI assistant. "
    "You always fulfill the user's requests to the best of your ability. "
    "Your answers must be concise, precise, and accurate."
)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_tokenizer = None
embedding_model = None
index = None
id_to_text = None
model = None

# Function to load FAISS index and text dictionary
def load_files(data_dir):
    global index, id_to_text
    # Search for files in the directory
    files = os.listdir(data_dir)
    
    # Filter FAISS and Pickle files
    faiss_file = [f for f in files if f.endswith('.index')]
    pickle_file = [f for f in files if f.endswith('.pkl')]
    
    # Validations
    if len(faiss_file) != 1:
        raise FileNotFoundError(f"Expected exactly one '.index' file in the directory '{data_dir}'.")
    
    if len(pickle_file) != 1:
        raise FileNotFoundError(f"Expected exactly one '.pkl' file in the directory '{data_dir}'.")

    # Load the FAISS index and the text dictionary
    index = faiss.read_index(os.path.join(data_dir, faiss_file[0]))
    with open(os.path.join(data_dir, pickle_file[0]), "rb") as f:
        id_to_text = pickle.load(f)

# Function to get normalized embeddings
def get_normalized_embeddings(texts):
    inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
    return embeddings

def get_related_docs(query, top_k=5):
    # Obtain normalized embeddings from the query text
    query_embedding = get_normalized_embeddings([query])
    
    # Search for the most similar documents in the FAISS index
    D, I = index.search(query_embedding, top_k)
    
    # Retrieve the texts of the most similar documents
    related_docs = [
        {
            "content": id_to_text[i]['content'],
            "file_name": id_to_text[i]['file_name'],
            "title": id_to_text[i]['title']
        } 
        for i in I[0]
    ]
    
    return related_docs

# Function to generate the model's response
def generate_response(prompt, system_prompt, chunks, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p):
    related = get_related_docs(prompt, chunks)
    formatted_related = "\n".join(
        [f"Related: {doc['content']}\nSource: {doc['file_name']} - {doc['title']}" for doc in related]
    )
    user_message = f"Question:{prompt}\nContext:{formatted_related}"
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
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
def gradio_interface(question, system_prompt, chunks, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p):
    # Generate response
    response = generate_response(question, system_prompt, chunks, temperature, max_tokens, top_k, repeat_penalty, min_p, top_p)
    # Display the question and response in the desired format
    return f"{response}"

def main():
    global embedding_tokenizer, embedding_model, model, device

    parser = argparse.ArgumentParser(description="Run a Gradio interface for a LLaMA-based question-answering model.")
    parser.add_argument('-d', '--data_dir', type=str, required=True, help="Directory containing the FAISS index and Pickle files.")
    
    args = parser.parse_args()

    # Detect if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Hugging Face model and tokenizer for embeddings
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)

    # Load the FAISS and Pickle files
    load_files(args.data_dir)

    # Load the Llama model
    model = Llama(model_path=model_path, 
                  n_gpu_layers=30 if torch.cuda.is_available() else 0, 
                  n_ctx=3584, 
                  n_batch=521, 
                  verbose=False, 
                  chat_format="llama-3")

    # Gradio Interface
    with gr.Blocks() as demo:
        # Always visible inputs
        question_input = gr.Textbox(label="Question")
        response_output = gr.Textbox(label="Output")
        
        # Collapsible section for settings
        with gr.Accordion("Advanced Settings", open=False):
            system_prompt_input = gr.Textbox(label="System Prompt", value=default_system_prompt, lines=5)
            chunks_slider = gr.Slider(1, 10, value=5, step=1, label="Chunks")
            temperature_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Temperature")
            max_tokens_slider = gr.Slider(-1, 512, value=-1, step=1, label="Max Tokens (use -1 for unlimited)")
            top_k_slider = gr.Slider(1, 100, value=40, step=1, label="Top K")
            repeat_penalty_slider = gr.Slider(1.0, 2.0, value=1.1, step=0.01, label="Repeat Penalty")
            min_p_slider = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Min P")
            top_p_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top P")

        # Submit button and logic
        generate_button = gr.Button("Generate Response")
        generate_button.click(
            fn=gradio_interface, 
            inputs=[
                question_input,
                system_prompt_input,
                chunks_slider,
                temperature_slider,
                max_tokens_slider,
                top_k_slider,
                repeat_penalty_slider,
                min_p_slider,
                top_p_slider,
            ], 
            outputs=response_output
        )

        # Submit on enter key
        question_input.submit(
            fn=gradio_interface, 
            inputs=[
                question_input,
                system_prompt_input,
                chunks_slider,
                temperature_slider,
                max_tokens_slider,
                top_k_slider,
                repeat_penalty_slider,
                min_p_slider,
                top_p_slider,
            ], 
            outputs=response_output
        )

    demo.launch()

if __name__ == "__main__":
    main()
