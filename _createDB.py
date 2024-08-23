import os
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np
import pickle
import argparse
from tqdm import tqdm

def get_normalized_embeddings(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Process text documents to create FAISS index and text embeddings.")
    parser.add_argument('-d', '--directory_path', type=str, required=True, help="Directory containing the text files.")
    parser.add_argument('-o', '--output_path', type=str, required=True, help="Directory to save the FAISS index and pickle file.")
    parser.add_argument('--chunk_size', type=int, default=1000, help="Size of text chunks in characters. Default is 1000.")
    parser.add_argument('--chunk_overlap', type=int, default=200, help="Overlap size between chunks in characters. Default is 200.")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Created output directory: {args.output_path}")

    # Detect if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Hugging Face model and tokenizer
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)  # Move the model to the GPU or CPU

    # Load all documents from the directory
    print("Reading documents...")
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(args.directory_path,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             loader_kwargs=text_loader_kwargs,
                             show_progress=True,
                             use_multithreading=True)
    documents = loader.load()

    # Create a splitter in terms of chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,  # Chunk size in characters
        chunk_overlap=args.chunk_overlap  # Overlap between chunks for additional context
    )

    # Split the documents into chunks
    print("Creating chunks...")
    split_docs = text_splitter.split_documents(documents)

    # Generate and normalize embeddings for all text chunks
    print("Generating embeddings...")
    all_embeddings = []
    id_to_text = {}
    for i, doc in tqdm(enumerate(split_docs), total=len(split_docs), desc="Processing documents"):
        embeddings = get_normalized_embeddings([doc.page_content], tokenizer, model, device)
        all_embeddings.append(embeddings)
        id_to_text[i] = doc.page_content  # Map the ID to the text

    # Convert the list of embeddings into a single numpy array
    all_embeddings = np.vstack(all_embeddings)

    # Create a FAISS index for inner product (for cosine similarity)
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product

    # Add embeddings to the FAISS index
    print("Adding embeddings to FAISS index...")
    index.add(all_embeddings)

    # Save the FAISS index to a file
    faiss.write_index(index, os.path.join(args.output_path, "faiss_index_cosine.index"))

    # Save the dictionary mapping IDs to texts
    with open(os.path.join(args.output_path, "id_to_text.pkl"), "wb") as f:
        pickle.dump(id_to_text, f)

    print(f"FAISS index and text dictionary successfully created and saved in the directory '{args.output_path}'.")

if __name__ == "__main__":
    main()
