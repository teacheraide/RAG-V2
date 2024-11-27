## Virtual Environment
It is strongly recommended to use a virtual environment to maintain an isolated environment for the project.

### Setting Up a Virtual Environment (Windows)

Create the virtual environment
  ```bash
  python -m venv venv
  ```

 Activate the virtual environment
  ```bash
  venv\Scripts\activate
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
  "prompt": "how big is brisbane",
  "temperature": 0.7,
  "max_tokens": 5000
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
### Note that max_tokens should be small (such as 500)
```bash
{
  "prompt": "how big is brisbane",
  "temperature": 0.7,
  "max_tokens": 500
}
```
