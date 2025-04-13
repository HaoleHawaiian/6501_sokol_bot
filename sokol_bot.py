# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:49:22 2025

@author: aloha
"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama_v1.1_math_code"  # or "TinyLlama/TinyLlama_v1.1" for standard
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit interface
st.title("OMSA's ISYE 6501 Chatbot S.O.K.O.L. (Student Oriented Knowledge for Online Learning")
user_input = st.text_input("Enter your question:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
