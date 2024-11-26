from flask import Flask, request, jsonify
import requests
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

folder_path = "db"

embedding = FastEmbedEmbeddings()

# Hugging Face API key and endpoint
api_key = "APIKEY"
hf_endpoint = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")

    data = request.get_json()

    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid request, 'prompt' field is required"}), 400
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": data['prompt'],  # You can customize this based on the model type
        "parameters": {
            "temperature": data.get("temperature", 0.7),
            "max_length": data.get("max_tokens", 1000)
        }
    }    

    try:
        response = requests.post(hf_endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()

        # Return the result back to the client
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf", methods=["POST"])
def pdfPost():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    if not file or file.filename == '':
        return {"error": "No valid file selected"}, 400

    file_name = file.filename
    os.makedirs("pdf", exist_ok=True)
    save_file = os.path.join("pdf", file_name)
    
    try:
        file.save(save_file)
        print(f"filename: {file_name}")

        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        print(f"docs len={len(docs)}")

        chunks = text_splitter.split_documents(docs)
        print(f"chunks len={len(chunks)}")

        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )
        vector_store.persist()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {"error": f"Processing error: {str(e)}"}, 500

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    data = request.get_json()

    # Validate request data
    prompt = data.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}, 400

    # Default values for parameters
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 500)

    try:
        # Load vector store
        print("Loading vector store")
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

        # Create retriever
        print("Creating retriever")
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,
                "score_threshold": 0.5,
            },
        )

        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(prompt)
        context = " ".join([doc.page_content for doc in docs])

        # If no relevant documents are found, return "no related answer"
        if not docs:
            return jsonify({"answer": "no related answer"})

        # Limit context length to prevent exceeding model token limits
        max_context_length = 1024  # Define a reasonable length for context
        context = context[:max_context_length]

        # Hugging Face API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": f"Context: {context}\nQuestion: {prompt}",
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": min(max_tokens, 4096 - len(context.split())),  # Ensure token limit safety
                "return_full_text": False
            },
        }

        print("Calling Hugging Face API")
        response = requests.post(hf_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Debugging output for Hugging Face API response
        print(f"HF API response: {result}")

        # Handle list or dictionary response structure
        if isinstance(result, list):
            result = result[0] if len(result) > 0 else {}

        # Extract the generated text safely
        generated_text = result.get("generated_text", "") if isinstance(result, dict) else ""

        # If the result is empty or not relevant, return "no related answer"
        if not generated_text or "irrelevant" in generated_text.lower():
            return jsonify({"answer": "no related answer"})

        # If relevant, combine the RAG answer with the original prompt
        combined_answer = f"Prompt: {prompt}\nAnswer: {generated_text}"

        # Prepare response with sources
        sources = [
            {"source": doc.metadata.get("source", "unknown"), "page_content": doc.page_content}
            for doc in docs
        ]

        return {"answer": combined_answer, "sources": sources}

    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return {"error": f"API request error: {str(e)}"}, 500
    except Exception as e:
        print(f"Error during processing: {e}")
        return {"error": f"Processing error: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)