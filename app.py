import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model_name = './fine_tuned_model'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

st.title("AI-Powered Story Generation for Kids")

# User input
user_input = st.text_input("Enter a story prompt:", "")

# Generate story
if st.button("Generate Story"):
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(story)
