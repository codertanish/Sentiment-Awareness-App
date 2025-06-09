#Uncomment the following lines to install the required packages
# !pip install transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import streamlit as st
@st.cache_resource
def load_sen_model():
    model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion")
    tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-emotion")
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
sentiment_pipeline = load_sen_model()  # Load the model and tokenizer

def generate_explanation(emotion):
    text = f"I feel {emotion}. Can you tell me more about this feeling, scientifically at a impersonal level?"
    inputs = tokenizer(text, return_tensors="pt")
    reply_ids = model.generate(**inputs,
                                max_length=400,
                                  do_sample=True,
                                    temperature=0.7,
                                      top_k=50,
                                      no_repeat_ngram_size=2)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

def generate_response(text, emotion):
    text = f"{text}. I am {emotion}"
    inputs = tokenizer(text, return_tensors="pt")
    reply_ids = model.generate(**inputs,
                                max_length=400,
                                  do_sample=True,
                                    temperature=0.7,
                                      top_k=50,
                                      no_repeat_ngram_size=2)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

st.title("Sentiment-Awareness App")
st.write("This app uses a pre-trained BERT model to perform sentiment analysis on text input. It also generates a response based on the detected sentiment using a BlenderBot model.")
st.subheader("Sentimental Text")
input_text = st.text_area("Enter text for sentiment analysis:", "Enter sentimental text here...")
if not input_text.strip():
    st.warning("Please enter some text to analyze.")
result = sentiment_pipeline(input_text)[0]["label"]
definition = generate_explanation(result)
if st.button("Analyze Sentiment and Generate Response"):
    st.subheader("Sentiment Result")
    text = f"The sentiment of the input text is: {result}. Can you tell me about {result}? {definition}"
    st.write(text)
    st.subheader("Response:")
    st.write(generate_response(text, result))


