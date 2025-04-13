import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import sys

# Fix for asyncio event loop on Windows
if sys.version_info >= (3, 8) and sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama_v1.1_math_code"
model = AutoModelForCausalLM.from_pretrained(model_name)
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
print(f"Pad token: {tokenizer.pad_token}, Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token}, EOS token ID: {tokenizer.eos_token_id}")
print(f"Vocabulary size: {len(tokenizer)}")

# Streamlit interface
st.title("OMSA's ISYE 6501 Chatbot S.O.K.O.L. (Student Oriented Knowledge for Online Learning)")
user_input = st.text_input("Enter your question:")

if user_input:
    # Tokenize input with padding and truncation
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        padding=True,  # Automatically pads sequences
        truncation=True,
        max_length=512  # Adjust max_length as needed
    )

    # Generate response from model
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=100,  # Set a maximum length for the output
        num_return_sequences=1,  # Generate one response
        no_repeat_ngram_size=2,  # Prevent repetition of n-grams
        temperature=0.7,  # Control randomness
        top_p=0.9,  # Nucleus sampling
        top_k=50,  # Top-k sampling
        do_sample=True  # Enable sampling for more diverse answers
    )
    
    # Decode response tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display response in Streamlit app
    st.write(response)
