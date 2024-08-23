from llama_cpp import Llama
from tqdm import tqdm
import time
import argparse
import os
import sys
import torch

# Path to the Llama model
model_path = "/path/to/llama3-8B-model.gguf"

# Global variables for the model and configuration
system_prompt = (
    "You are a helpful, smart, kind, and efficient AI assistant. "
    "You always fulfill the user's requests to the best of your ability. "
    "Your answers must be concise, precise, and accurate."
)
temperature = 0.7
max_tokens = -1
top_k = 40
repeat_penalty = 1.1
min_p = 0.05
top_p = 0.95


# Detect if CUDA is available and configure the model accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the Llama model with GPU support if available
model = Llama(
    model_path=model_path, 
    n_gpu_layers=30 if device == "cuda" else 0,  # Use GPU layers if CUDA is available
    n_ctx=3584, 
    n_batch=521, 
    verbose=False, 
    chat_format="llama-3"
)

# Function to generate responses
def generate_response(prompt, model):
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

def main():
    global system_prompt

    parser = argparse.ArgumentParser(description="Process questions using a language model.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file with questions (e.g., questions.txt)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file to save the answers (e.g., answers.txt)")
    parser.add_argument('-p', '--passes', type=int, default=1, help="Number of passes over the questions. Default is 1.")

    args = parser.parse_args()

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
                        answer = generate_response(question, model)
                        answer_file.write(f"Q: {question}\nA: {answer}\n\n")
    finally:
        # Ensure the Llama model is properly closed
        model.__del__()

    # Total time at the end
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time: {total_time:.2f} seconds")
    print(f"All passes completed. Answers saved to '{args.output}' with appropriate suffixes.")

if __name__ == "__main__":
    main()
