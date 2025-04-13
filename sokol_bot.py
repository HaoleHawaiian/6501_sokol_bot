# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:49:22 2025

@author: aloha
"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama_v1.1_math_code"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check and set pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Explicitly add a [PAD] token

# Resize model embeddings to account for added special tokens
model.resize_token_embeddings(len(tokenizer))


# Streamlit interface
st.title("OMSA's ISYE 6501 Chatbot S.O.K.O.L. (Student Oriented Knowledge for Online Learning)")
user_input = st.text_input("Enter your question:")

if user_input:
    # Tokenize the input with padding and truncation
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        padding=True,  # Automatically pads to the longest sequence in the batch
        truncation=True,
        max_length=512  # Adjust max_length as needed
    )

    
    # Generate the model's response
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
    
    # Decode the output tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Show the response
    st.write(response)