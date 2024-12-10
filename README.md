# Quick start

To run this project you will need a Hugging Face API Key, create one [here](https://huggingface.co/settings/tokens)

a `Read` token is enough to run this project

## Start server with docker compose

copy `.env.sample` to `.env` and add your Hugging Face API Key

```
docker compose up
```

# Development

## Virtual Environment

It is strongly recommended to use a virtual environment to maintain an isolated environment for the project.

### Setting Up a Virtual Environment (Windows)

Create the virtual environment

```bash
python -m venv venv
```

Activate the virtual environment
windows:

```bash
venv\Scripts\activate
```

mac / \*inx:

```bash
source venv/bin/activate
```

## Requirements

- **Python Version**: Ensure you are using **Python 3.10**. You can download this version from Python official website and add it to the PATH when installing.
- **Dependencies**: Install all dependencies listed in `requirements.txt` using the command:

  ```bash
  pip install -r requirements.txt
  ```

## Routes Overview

The Flask app provides the following three routes:

### 1. /ai

A basic route that interacts with the Hugging Face API to generate answers.

Request Format:

```bash
{
    "chat_id": "12345",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}
```

### 2. /pdf

This route processes a PDF file by chunking it into smaller parts and storing these chunks in the folder db.

Key: file
<br> Value: Name of the PDF file (e.g., PrinciplesOfBiology.pdf)
<br> The generated vector_store stored in the db folder will be used in the /ask_pdf route.

### 3. /ask_pdf

This route combines Retrieval-Augmented Generation (RAG) and an LLM model via the Hugging Face API.

Request Format:

### Note that max_tokens should be small (such as 2000)

```bash
{
    "chat_id": "12345",
    "messages": [
        {"role": "user", "content": "Where is Prophase?"}
    ],
    "temperature": 0.7,
    "max_tokens": 3000
}
```
