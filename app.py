import os
import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Existing imports
from pathlib import Path
import streamlit as st
import tiktoken
import torch
from src.gpt import GPTModel
from src.utils.train import generate, text_to_token_ids, token_ids_to_text

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_model_and_tokenizer():
    GPT_CONFIG_355M = {
        "vocab_size": 50257,    
        "context_length": 1024,  
        "emb_dim": 1024,        
        "n_heads": 16,       
        "n_layers": 24,      
        "drop_rate": 0.0,       
        "qkv_bias": True     
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path('models') / 'gpt2-medium355M-sft-standalone.pth'
    if not model_path.exists():
        st.error(f"Could not find the {model_path} file.")
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_355M

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

# Load model and tokenizer
tokenizer, model, model_config = get_model_and_tokenizer()

# UI
st.title("Chatbot")
st.write("Interact with the tuned GPT-2 model in a chatbot interface.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format prompt for GPT-2
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {prompt}
    """

    # Set seed for reproducibility
    torch.manual_seed(123)

    # Generate token IDs
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(formatted_prompt, tokenizer).to(device),
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    # Convert token IDs to text
    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, formatted_prompt)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)