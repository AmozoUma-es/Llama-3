import faiss
import os
import pickle
import torch
from tqdm import tqdm
import time
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModel
import argparse
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Solves OpenMP error

# Path to the Llama model
model_path = "/path/to/llama3-8B-model.gguf"

# Global variables for the model, tokenizer, index, and text dictionary
system_prompt = (
    "You are a helpful, smart, kind, and efficient AI assistant. "
    "You always fulfill the user's requests to the best of your ability. "
    "Your answers must be concise, precise, and accurate."
)
embedding_tokenizer = None
embedding_model = None
index = None
id_to_text = None
llama_model = None
temperature = 0.7
max_tokens = -1
top_k = 40
repeat_penalty = 1.1
min_p = 0.05
top_p = 0.95
chunks = 5  # how many related documents chunks to add to the prompt

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
    related_docs = [id_to_text[i] for i in I[0]]
    
    return related_docs

def generate_response(prompt):
    related = get_related_docs(prompt, chunks)
    formatted_related = "\n".join([f"Related: {doc}" for doc in related])
    user_message = f"Question:{prompt}\nContext:{formatted_related}"
    response = llama_model.create_chat_completion(
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

def main():
    global embedding_tokenizer, embedding_model, llama_model, system_prompt, device

    parser = argparse.ArgumentParser(description="Process questions using a language model.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file with questions (e.g., questions.txt)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file to save the answers (e.g., answers.txt)")
    parser.add_argument('-d', '--data_dir', type=str, required=True, help="Directory containing the FAISS (.index) and Pickle (.pkl) files")
    parser.add_argument('-p', '--passes', type=int, default=1, help="Number of passes over the questions. Default is 1.")

    args = parser.parse_args()

    # Detect if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Hugging Face model and tokenizer for embeddings
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)

    # Load the FAISS and Pickle files
    try:
        load_files(args.data_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Load the Llama model
    llama_model = Llama(model_path=model_path, 
                        n_gpu_layers=30 if torch.cuda.is_available() else 0, 
                        n_ctx=3584, 
                        n_batch=521, 
                        verbose=False, 
                        chat_format="llama-3")

    # Read the questions from the file specified in the arguments
    with open(args.input, "r", encoding="utf-8") as question_file:
        questions = question_file.readlines()

    # Initialize total time
    start_time = time.time()

    try:
        # Perform the specified number of passes
        for pass_num in range(1, args.passes + 1):
            output_filename = f"{os.path.splitext(args.output)[0]}-{pass_num}{os.path.splitext(args.output)[1]}"
            print(f"Processing pass {pass_num}, saving to {output_filename}")

            # Open a file to write the answers for this pass
            with open(output_filename, "w", encoding="utf-8") as answer_file:
                for question in tqdm(questions, desc=f"Processing questions (pass {pass_num})"):
                    question = question.strip()  # Remove surrounding whitespace
                    if question:  # Ensure the line is not empty
                        answer = generate_response(question)
                        answer_file.write(f"Q: {question}\nA: {answer}\n\n")
    finally:
        # Ensure the Llama model is properly closed
        llama_model.__del__()

    # Total time at the end
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time: {total_time:.2f} seconds")
    print(f"All passes completed. Answers saved to '{args.output}' with appropriate suffixes.")

if __name__ == "__main__":
    main()
