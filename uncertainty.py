import streamlit as st
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

class CausalityAnalyzer:
    def __init__(self, model_path, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if os.path.exists(self.model_path):
            # Load fine-tuned model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            # Load pre-trained model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2
            )
        self.model.to(self.device)

    def train_model(self, train_file, val_file, test_file, epochs=5):
        # Load data
        train_df = pd.read_csv(train_file, encoding='UTF-8')
        val_df = pd.read_csv(val_file, encoding='UTF-8')
        test_df = pd.read_csv(test_file, encoding='UTF-8')

        # Convert to datasets
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df)
        })

        # Tokenize
        def tokenize(batch):
            return self.tokenizer(batch["sentence"], padding=True, truncation=True)

        dataset_encoded = dataset.map(tokenize, batched=True)
        dataset_encoded = dataset_encoded.remove_columns(['sentence'])
        dataset_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # Training arguments
        from transformers import TrainingArguments, Trainer

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            f1 = f1_score(labels, preds, average="weighted")
            acc = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average="weighted")
            recall = recall_score(labels, preds, average="weighted")
            return {
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_encoded["train"],
            eval_dataset=dataset_encoded["validation"],
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.model_path)

    def predict(self, text):
        if len(text) <= 500:
            inputs = self.tokenizer(
                text,
                padding='longest',
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = np.argmax(outputs.logits.cpu().numpy(), axis=1)[0]
            confidence = probabilities[0][prediction].item()
            
            return prediction, confidence
        return 0, 0.0

def main():
    st.set_page_config(page_title="Uncertainty Analyzer", layout="wide")
    
    st.title("Uncertainty Analyzer")
    
    # Sidebar for model selection and training
    st.sidebar.header("Model Configuration")
    
    model_options = {
        "English": "bert-base-uncased",
        "Japanese": "cl-tohoku/bert-base-japanese-whole-word-masking"
    }
    
    selected_language = st.sidebar.selectbox(
        "Select Language Model",
        options=list(model_options.keys())
    )
    
    model_name = model_options[selected_language]
    model_path = "./models/causality_model"
    
    # Initialize analyzer
    analyzer = CausalityAnalyzer(model_path, model_name)
    
    # Training section in sidebar
    st.sidebar.header("Model Fine-tuning")
    train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type="csv")
    val_file = st.sidebar.file_uploader("Upload Validation Data (CSV)", type="csv")
    test_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type="csv")
    
    epochs = st.sidebar.slider("Training Epochs", 1, 10, 5)
    
    if st.sidebar.button("Train Model"):
        if train_file and val_file and test_file:
            with st.spinner("Training model..."):
                analyzer.load_model()
                analyzer.train_model(train_file, val_file, test_file, epochs)
                st.success("Model training completed!")
        else:
            st.error("Please upload all required training files.")
    
    # Main area for text analysis
    st.header("Text Analysis")
    
    # Load the model for prediction
    try:
        analyzer.load_model()
    except Exception as e:
        st.warning("No trained model found. Please train a model first.")
        return
    
    text_input = st.text_area(
        "Enter text to analyze",
        height=200,
        max_chars=1000,
        help="Enter the text you want to analyze for causality (max 500 characters)"
    )
    
    if st.button("Analyze"):
        if text_input:
            with st.spinner("Analyzing text..."):
                prediction, confidence = analyzer.predict(text_input)
                
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

if __name__ == "__main__":
    main()