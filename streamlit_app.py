import streamlit as st
import json
import torch




import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
from transformers import GPT2LMHeadModel, GPT2Tokenizer



tok = GPT2Tokenizer.from_pretrained("content/weights_push_v_may")
model = GPT2LMHeadModel.from_pretrained("content/weights_push_v_may")


def push(text,lenght=100,temp=1,num=2):
  
  repetition_penalty = 2.6
  temperature = temp
  top_k =4 

  inpt = tok.encode(text, return_tensors="pt")

  max_length=lenght

  out = model.generate(inpt,max_length= max_length, 
                       repetition_penalty=repetition_penalty, 
                       do_sample=True, top_k=top_k, top_p=0.95, 
                       temperature=temperature,num_return_sequences=num)
  decoded = tok.decode(out[0])
  return decoded


#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Neural pushkin, created by Alex Wortega(https://t.me/datapron), Arina Pushkova(https://t.me/def_model_train)"

st.title('Neural pushkin generator')
st.write(desc)

tmp= st.number_input('Temperature', min_value=0.0, max_value=3.0, value=0.1)
user_input = st.text_input('Text promt:')


if st.button('Generate Text'):
    generated_text = push(user_input,temp=tmp)

    st.write(generated_text)
