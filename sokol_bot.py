import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import sys
import torch

# Fix for asyncio event loop on Windows
if sys.version_info >= (3, 10):
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError as e:
        print(f"AsyncIO setup error: {e}")

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama_v1.1_math_code"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad_token is properly set
if tokenizer.pad_token is None:
    try:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add [PAD] token explicitly
        print("Added [PAD] token.")
    except Exception as e:
        print(f"Error adding [PAD] token: {e}")
        tokenizer.pad_token = tokenizer.eos_token  # Fallback to eos_token

# Resize model embeddings to account for added special tokens
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# Debugging output for verification
# print(f"Pad token: {tokenizer.pad_token}, Pad token ID: {tokenizer.pad_token_id}")
# print(f"EOS token: {tokenizer.eos_token}, EOS token ID: {tokenizer.eos_token_id}")
# print(f"Vocabulary size: {len(tokenizer)}")

# Streamlit interface
st.title("OMSA's ISYE 6501 Chatbot S.O.K.O.L. (Student Oriented Knowledge for Online Learning)")
user_input = st.text_input("Enter your question:")

if user_input:
    # Tokenize input with padding and truncation
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    outputs = model.generate(
        inputs['input_ids'],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True
    )
    
    # Decode response tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display response in Streamlit app
    st.write(response)
