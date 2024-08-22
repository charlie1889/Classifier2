import streamlit as st
import pandas as pd
import torch
import pickle
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(output_dir):
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

# Load the MultiLabelBinarizer
@st.cache_resource
def load_mlb(mlb_file):
    with open(mlb_file, 'rb') as f:
        mlb = pickle.load(f)
    return mlb

# Preprocess text
def preprocess_text(text, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding

# Predict labels
def predict(text, model, tokenizer, mlb, device, max_len=128):
    model.eval()  # Set the model to evaluation mode
    encoding = preprocess_text(text, tokenizer, max_len)
    input_ids = encoding['input_ids'].to(device)  # Move to device
    attention_mask = encoding['attention_mask'].to(device)  # Move to device

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()

    threshold = 0.5
    predictions = (predictions >= threshold).astype(int)
    predicted_labels = mlb.inverse_transform(predictions)
    formatted_labels = '/'.join(label for label in predicted_labels[0])
    return formatted_labels

# Streamlit app
def main():
    st.title("Text Classification with BERT")
    st.write("Upload a CSV file with a 'sentence' column to classify.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the CSV
        df = pd.read_csv(uploaded_file)
        df = df.dropna(subset=['sentence'])

        # Load the MultiLabelBinarizer
        mlb_file = 'mlb.pkl'  # Path to the saved MultiLabelBinarizer
        mlb = load_mlb(mlb_file)

        # Load the model and tokenizer
        output_dir = './saved_model'  # Replace with your model directory
        model, tokenizer, device = load_model_and_tokenizer(output_dir)

        # Display a loader while processing the input
        with st.spinner('Processing...'):
            # Iterate through each sentence in the CSV and predict
            predicted_labels_list = []
            for index, row in df.iterrows():
                sentence = row['sentence']
                predicted_labels = predict(sentence, model, tokenizer, mlb, device)
                predicted_labels_list.append(predicted_labels)

            # Add the predictions to the DataFrame
            df['predicted_labels'] = predicted_labels_list

        # Display the DataFrame
        st.success('Processing complete!')
        st.write(df)

        # Optionally download the DataFrame as a new CSV file
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predicted_labels.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
