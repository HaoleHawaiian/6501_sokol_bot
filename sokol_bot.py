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

# # Resize model embeddings to account for added special tokens
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Streamlit interface
st.title("OMSA's ISYE 6501 Chatbot S.O.K.O.L. (Student Oriented Knowledge for Online Learning)")
user_input = st.text_input("Enter your question:")

if user_input:
    # Tokenize input with padding and truncation
    prompt = f"""You are a helpful tutor for Georgia Tech's ISYE 6501 course. Answer questions clearly and concisely.

User: {user_input}
Assistant:"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate response using the model
    with torch.no_grad():  # Disable gradients for inference
        outputs = model.generate(
            inputs['input_ids'],  # Provide the input IDs
            max_length=100,  # Limit the output length
            num_return_sequences=1,  # Generate one sequence
            no_repeat_ngram_size=2,  # Avoid repetition
            temperature=0.7,  # Sampling temperature
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Top K sampling
            do_sample=True  # Enable sampling (not greedy)
        )
    
    # Decode response tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (after "Assistant:")
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    # Display response in Streamlit app
    st.write(response)
