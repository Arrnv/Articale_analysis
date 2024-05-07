# import streamlit as st
# from transformers import AddedToken

# # Define a custom hash function for tokenizers.AddedToken
# def my_hash_func(token):
#     try:
#         return hash((token.ids, token.type_id))
#     except AttributeError:
#         # Handle cases where the token object is not as expected
#         return hash(str(token))


# my_module.py
import streamlit as st
# from transformers import AddedToken

# # Define a custom hash function for tokenizers.AddedToken
# def my_hash_func(token):
#     try:
#         return hash((token.ids, token.type_id))
#     except AttributeError:
#         # Handle cases where the token object is not as expected
#         return hash(str(token))

from setup import analyzer, emotion_analyzer, hate_speech_analyzer
@st.cache_resource()
def get_analyzers():
    
    return analyzer, emotion_analyzer, hate_speech_analyzer
