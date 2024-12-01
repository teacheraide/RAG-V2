from flask import Flask, request, jsonify, session
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
app.secret_key = 'teacheraide'  # Required for session handling

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

    # Validate input data
    data = request.get_json()
    if not data or "messages" not in data or "chat_id" not in data:
        return jsonify({"error": "Invalid request, 'messages' and 'chat_id' fields are required"}), 400

    # Retrieve chat_id
    chat_id = data["chat_id"]
    print(f"Processing chat ID: {chat_id}")

    # Restrict to the last user message
    last_message = data["messages"][-1]["content"]

    conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in data["messages"]])
    hf_payload = {
        "inputs": f"{conversation}\nAssistant:",
        "parameters": {
            "temperature": data.get("temperature", 0.3),
            "max_new_tokens": data.get("max_tokens", 500),
            "stop_sequence": ["\nUser:"],  # Prevent the model from continuing user messages
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Make API call to Hugging Face
        response = requests.post(hf_endpoint, headers=headers, json=hf_payload)
        response.raise_for_status()
        hf_result = response.json()

        # Extract response content
        generated_text = hf_result[0].get("generated_text", "") if isinstance(hf_result, list) else hf_result.get("generated_text", "")

        # Format response to mimic OpenAI's structure
        openai_response = {
            "chat_id": chat_id,  # Include chat_id in the response
            "id": "hf-chat",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": generated_text.strip()},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(last_message.split()) + len(generated_text.split())
            }
        }

        return jsonify(openai_response)

    except requests.exceptions.RequestException as e:
        return jsonify({"chat_id": chat_id, "error": f"API request error: {str(e)}"}), 500
    except KeyError as e:
        return jsonify({"chat_id": chat_id, "error": f"Response parsing error: {str(e)}"}), 500


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
    if not data or "messages" not in data or "chat_id" not in data:
        return jsonify({"error": "Invalid request, 'messages' and 'chat_id' fields are required"}), 400

    chat_id = data["chat_id"]
    prompt = data["messages"][-1]["content"]  # Get the last user message
    print(f"Processing chat ID: {chat_id}, prompt: {prompt}")

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
            return jsonify({"chat_id": chat_id, "error": "No related answer found"}), 400

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
            return jsonify({"chat_id": chat_id, "error": "No relevant answer found"}), 400

        # Format the response to mimic OpenAI's structure
        openai_response = {
            "chat_id": chat_id,  # Include chat_id in the response
            "id": "hf-chat",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": generated_text.strip()},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        }

        # Prepare sources (optional)
        sources = [
            {"source": doc.metadata.get("source", "unknown"), "page_content": doc.page_content}
            for doc in docs
        ]

        return jsonify({"answer": generated_text, "sources": sources, "openai_response": openai_response})

    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return jsonify({"chat_id": chat_id, "error": f"API request error: {str(e)}"}), 500
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"chat_id": chat_id, "error": f"Processing error: {str(e)}"}), 500
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
