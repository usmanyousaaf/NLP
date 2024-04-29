import streamlit as st
import wikipediaapi  
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Cache the loading of models and tokenizers to speed up the process
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def get_relevant_content(question, results):
    # Get the top 3 search results and extract relevant sections
    content = ""
    for result in results[:3]:
        try:
            page = wiki.page(result)
            content += page.summary  # Using summary to get concise content
        except wiki.exceptions.DisambiguationError:
            continue  # If there's a disambiguation error, skip it and continue with the next one
    return content

def main():
    st.title("Chat Bot App")
    st.image("chatbot.png", width=90)  # Assuming you have an image file named chatbot.png in your app directory

    # Create a sidebar for question history
    st.sidebar.title("Question History")
    question_history = st.sidebar.empty()

    # Initialize the question history list
    if 'question_history' not in st.session_state:
        st.session_state.question_history = []

    # Input box for user questions
    ques = st.text_input("Ask your question:")

    # Style the buttons for BERT and Roberta
    st.write("Choose a model:")
    col1, col2 = st.columns(2)
    with col1:
        button_bert = st.button("BERT", key="bert_button")
    with col2:
        button_roberta = st.button("Roberta", key="roberta_button")

    # Load the selected model and tokenizer
    if button_roberta:
        model_name = "roberta-large"
    else:
        model_name = "bert-large-uncased"

    tokenizer, model = load_model_and_tokenizer(model_name)

    # Check if a question is asked and at least one model is selected
    if ques and (button_bert or button_roberta):
        st.session_state.question_history.append(ques)
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(ques)
        text = page.text

        inputs = tokenizer.encode_plus(ques, text, return_tensors='pt', max_length=512, truncation=True)
        answer_start_scores, answer_end_scores = model(**inputs).values()

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        ans = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end], skip_special_tokens=True)

        st.write(f"Question: {ques}")
        st.write(f"Answer: {ans}")

    # Display question history
    question_history.write("Question History:")
    for i, question in enumerate(st.session_state.question_history):
        question_history.write(f"{i + 1}. {question}")

if __name__ == "__main__":
    main()
    