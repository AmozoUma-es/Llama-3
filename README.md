# Llama-3

A set of files to use Llama 3 models in different ways. I made my test using Llama 3 8B (8bit) Instruct on a laptop with a 4070 8GB.

## Project Overview

This repository contains scripts to utilize the LLaMA 3 model for various tasks, including direct model usage and Retrieval-Augmented Generation (RAG). There are scripts for batch processing as well as web-based interaction using Gradio.

All scripts are fully independent, you can copy and use a anyone without the others.

You have to update the model_path in the scripts pointing to the gguf file in your system.

Installation instructions at the bottom.

### **Direct Model Usage**

The scripts `llama-cpp.py` and `llama-cpp-gradio.py` allow you to interact directly with the LLaMA 3 model.

- **Batch Processing (llama-cpp.py):**
  - This script processes a batch of input questions and generates responses using the LLaMA model.
  - Parameters:
    - `--input`: Path to the input file containing questions.
    - `--output`: Path to the output file to save responses.
    - `--passes`: Number of passes over the input questions (optional).

- **Web Interface (llama-cpp-gradio.py):**
  - This script provides a web-based interface using Gradio to interact with the LLaMA model.
  - You can adjust model parameters like temperature, max tokens, top_k, etc., through the Gradio interface.
  - Does not use contaxt, each question is independent 

### **Streaming Chat Interface**

The `llama-cpp-gradio-chat.py` script offers a real-time chat interface with streaming responses, allowing for interactive conversations with the LLaMA model.

- **Streaming Chat with Gradio (llama-cpp-gradio-chat.py):**
  - Provides a live chat interface using Gradio.
  - Features settings for adjusting system prompt, temperature, max tokens, and other model parameters.
  - Support context, the model 'remember' the prevous questions. Click on 'Clear Chat' to start a new session.
  - Supports streaming responses, where the user can see the output as it is being generated.

### **Using RAG (Retrieval-Augmented Generation)**

The scripts `llama-RAG.py` and `llama-RAG-gradio.py` extend the LLaMA model's capabilities by integrating it with a FAISS vector database to perform Retrieval-Augmented Generation.

- **Database Creation (createDB.py):**
  - Before using RAG, you must create a vector database using this script.
  - Parameters:
    - `--directory_path`: Path to the directory containing text files.
    - `--output_path`: Directory where the FAISS index and text embeddings will be saved.
    - `--chunk_size`: Size of text chunks in characters (optional).
    - `--chunk_overlap`: Overlap size between text chunks (optional).

- **Batch Processing with RAG (llama-RAG.py):**
  - This script processes a batch of input questions, retrieves relevant documents from the vector database, and generates responses.
  - Parameters:
    - `--input`: Path to the input file containing questions.
    - `--output`: Path to the output file to save responses.
    - `--data_dir`: Directory containing the FAISS index and Pickle files.
    - `--passes`: Number of passes over the input questions (optional).

- **Web Interface with RAG (llama-RAG-gradio.py):**
  - This script provides a web-based interface using Gradio for RAG. It retrieves relevant documents from the vector database and generates responses based on them.
  - You can adjust model parameters and the number of chunks to retrieve through the Gradio interface.


## Install requirements

```bash
pip install -r requirements.txt
```

You also need to install PyTorch and llama-cpp-python. You can install both with GPU support, check at the bottom of the content. If you don't have GPU just install both:

```bash
pip install torch llama-cpp-python
```

# Installation Guide: PyTorch and llama-cpp-python with GPU Support

This README provides a quick guide to install PyTorch and llama-cpp-python with GPU support.

## Prerequisites

Before starting, ensure you have the following:

- A compatible operating system (Linux, macOS, Windows).
- Python 3.10 or higher.
- CUDA 11.7 or higher installed (for GPU support in PyTorch).
- CUDA 12.1 or higher installed (for GPU support in llama-cpp-python).
- Access to the terminal or command line interface.

## Installing PyTorch

### 1. CUDA Installation

To enable GPU support in PyTorch, you need to have CUDA installed. Follow the instructions provided in the [official NVIDIA documentation](https://developer.nvidia.com/cuda-downloads).

### 2. Install PyTorch with GPU Support

You can install PyTorch with GPU support using the following command. Make sure to select the correct CUDA version that matches your installation.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with the appropriate CUDA version, such as `cu124` for CUDA 12.4.

For more details on PyTorch installation, refer to the [official installation guide](https://pytorch.org/get-started/locally/).

## Installing llama-cpp-python

### 1. Additional Requirements

To install llama-cpp-python with GPU support, ensure you have:

- CMake 3.18 or higher.
- A compiler compatible with C++17.

### 2. Install llama-cpp-python

You can install llama-cpp-python using pip:

```bash
pip install llama-cpp-python
```

### 3. Enabling GPU Support in llama-cpp-python

To enable GPU support, make sure the package is compiled with the appropriate flags for CUDA. This may require additional configuration in your development environment.

You can install llama-cpp-python with GPU support using the following command. Make sure to select the correct CUDA version that matches your installation.

```bash
set CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --upgrade --no-cache-dir --verbose
```

Replace `cu124` with the appropriate CUDA version, such as `cu124` for CUDA 12.4.

For detailed instructions on installation and configuration of llama-cpp-python, refer to the [official documentation](https://llama-cpp-python.readthedocs.io/en/stable/).

Maybe you should install a prevous version of numpy as langchain and faiss does not support numpy 2

```bash
pip install numpy==1.26
```

Some issues in Windows can be fixed copying some files: https://stackoverflow.com/a/56665992

## PyTorch Installation Verification

To verify that both libraries have been installed correctly and GPU support is enabled, run the following commands in a Python environment:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

If `torch.cuda.is_available()` returns `True`, PyTorch has been installed correctly with GPU support.

## Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/get-started/locally/)
- [llama-cpp-python Official Documentation](https://llama-cpp-python.readthedocs.io/en/stable/)

## Contributions

If you have suggestions or improvements for this README, feel free to open an issue or a pull request.
