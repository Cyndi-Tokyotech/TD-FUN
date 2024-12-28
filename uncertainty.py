import streamlit as st
import torch
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import accelerate
import transformers

class CausalityAnalyzer:
    def __init__(self, model_path, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load either fine-tuned model from path or pre-trained model"""
        if os.path.exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2
            )
        self.model.to(self.device)

    def prepare_dataset(self, df):
        """Convert DataFrame to HuggingFace Dataset"""
        dataset = Dataset.from_pandas(df)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
        return dataset

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted')
        }

    def train_model(self, train_df, validation_df, test_df, batch_size=16, epochs=3):
        """Train the model using provided datasets"""
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        validation_dataset = self.prepare_dataset(validation_df)
        test_dataset = self.prepare_dataset(test_df)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_path)
        
        return test_results

    def predict_batch(self, texts, batch_size=32):
        """Predict uncertainty for a batch of texts"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = np.argmax(outputs.logits.cpu().numpy(), axis=1)
            confidences = [probabilities[i][pred].item() for i, pred in enumerate(predictions)]
            
            batch_results = list(zip(predictions, confidences))
            results.extend(batch_results)
        
        return results

def main():
    st.set_page_config(page_title="Uncertainty Analyzer", layout="wide")
    
    st.title("Uncertainty Analyzer")
    
    # Sidebar for model selection and configuration
    st.sidebar.header("Model Configuration")
    
    model_options = {
        "English": "bert-base-uncased",
        "Chinese": "google-bert/bert-base-chinese",
        "Japanese": "cl-tohoku/bert-base-japanese-whole-word-masking"
    }
    
    selected_language = st.sidebar.selectbox(
        "Select Language Model",
        options=list(model_options.keys())
    )
    
    model_name = model_options[selected_language]
    model_path = f"./models/uncertainty_model_{selected_language.lower()}"
    
    # Initialize analyzer
    analyzer = CausalityAnalyzer(model_path, model_name)
    
    # Add tabs for training, single text analysis, and batch processing
    tab1, tab2, tab3 = st.tabs(["Model Training", "Single Text Analysis", "Batch Processing"])
    
    with tab1:
        st.header("Model Training")
        st.write("Upload your training, validation, and test datasets (CSV files with 'text' and 'label' columns)")
        
        train_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
        val_file = st.file_uploader("Upload Validation Dataset", type="csv", key="val")
        test_file = st.file_uploader("Upload Test Dataset", type="csv", key="test")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, value=16)
        with col2:
            epochs = st.number_input("Number of Epochs", min_value=1, value=3)
        
        if st.button("Train Model"):
            if train_file and val_file and test_file:
                try:
                    with st.spinner("Training model..."):
                        # Load datasets
                        train_df = pd.read_csv(train_file)
                        val_df = pd.read_csv(val_file)
                        test_df = pd.read_csv(test_file)
                        
                        # Verify columns
                        required_columns = {'text', 'label'}
                        for df, name in [(train_df, 'training'), (val_df, 'validation'), (test_df, 'test')]:
                            if not required_columns.issubset(df.columns):
                                st.error(f"Missing required columns in {name} dataset. Required: {required_columns}")
                                return
                        
                        # Load model for training
                        analyzer.load_model()
                        
                        # Train model
                        test_results = analyzer.train_model(
                            train_df, 
                            val_df, 
                            test_df, 
                            batch_size=batch_size,
                            epochs=epochs
                        )
                        
                        # Display results
                        st.success("Model training completed!")
                        st.write("### Test Set Results")
                        st.json(test_results)
                        
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            else:
                st.error("Please upload all required datasets.")
    
    with tab2:
        st.header("Single Text Analysis")
        
        # Load the model for prediction
        try:
            analyzer.load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        text_input = st.text_area(
            "Enter text to analyze",
            height=200,
            max_chars=1000,
            help="Enter the text you want to analyze for uncertainty (max 1000 characters)"
        )
        
        if st.button("Analyze Text"):
            if text_input:
                with st.spinner("Analyzing text..."):
                    results = analyzer.predict_batch([text_input])
                    prediction, confidence = results[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Prediction",
                            "Uncertain" if prediction == 1 else "Non-uncertain"
                        )
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{confidence:.2%}"
                        )
                    
                    # Visualization
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Bar(
                        x=['Non-uncertain', 'Uncertain'],
                        y=[1-confidence, confidence] if prediction == 1 else [confidence, 1-confidence],
                        marker_color=['#FF9999', '#99FF99']
                    ))
                    
                    fig.update_layout(
                        title="Prediction Confidence",
                        yaxis_title="Probability",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig)
            else:
                st.error("Please enter some text to analyze.")
    
    with tab3:
        st.header("Batch Processing")
        st.write("Upload a CSV file with a 'text' column containing the texts to analyze")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch")
        
        if uploaded_file is not None:
            if st.button("Process File"):
                try:
                    # Load the model
                    analyzer.load_model()
                    
                    with st.spinner("Processing file..."):
                        # Read and process file
                        df = pd.read_csv(uploaded_file)
                        if 'text' not in df.columns:
                            st.error("CSV file must contain a 'text' column")
                            return
                        
                        texts = df['text'].tolist()
                        results = analyzer.predict_batch(texts)
                        
                        df['prediction'] = [r[0] for r in results]
                        df['confidence'] = [r[1] for r in results]
                        df['uncertainty'] = df['prediction'].map({1: 'Uncertain', 0: 'Non-uncertain'})
                        
                        # Display results
                        st.write("### Results Preview")
                        st.dataframe(df.head())
                        
                        # Download button for results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="uncertainty_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Summary statistics
                        st.write("### Summary Statistics")
                        total = len(df)
                        uncertain = (df['prediction'] == 1).sum()
                        certain = total - uncertain
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Texts", total)
                        with col2:
                            st.metric("Uncertain Texts", uncertain)
                        with col3:
                            st.metric("Non-uncertain Texts", certain)
                        
                        # Visualization
                        import plotly.express as px
                        
                        fig = px.pie(
                            values=[certain, uncertain],
                            names=['Non-uncertain', 'Uncertain'],
                            title="Distribution of Predictions"
                        )
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()