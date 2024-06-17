import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def generate_story(model, tokenizer, prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

st.title("AI-Powered Story Generator for Kids")
st.subheader("Input your characters, setting, and theme:")
characters = st.text_input("Characters")
setting = st.text_input("Setting")
theme = st.text_input("Theme")

if st.button("Generate Story"):
    prompt = f"{characters} in {setting} with a theme of {theme}"
    tokenizer, model = load_model("./fine_tuned_model")
    story = generate_story(model, tokenizer, prompt)
    st.write(story)
