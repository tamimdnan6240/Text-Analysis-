#!/usr/bin/env python
# coding: utf-8

# Explanation of Each Code Line
# 
# Imports: Import necessary libraries for deep learning and data handling.
# 
# Initialization: Set up lists to store training and validation losses.
# 
# Epoch Loop: Loop over the number of epochs specified by EPOCHS.
# 
# Training Mode: Set the model to training mode.
# 
# Initialize Metrics: Initialize variables to track loss and accuracy.
# 
# Batch Loop: Loop over batches in the training dataloader.
# 
# Data to Device: Move input data to the GPU or CPU.
# 
# Zero Gradients: Clear previous gradients.
# 
# Forward Pass: Get model outputs.
# 
# Calculate Loss: Compute loss using the criterion.
# 
# Accumulate Loss: Add the loss to the total training loss.
# 
# Predictions: Get predictions by finding the index of the maximum logit.
# 
# Count Correct Predictions: Increment the count of correct predictions.
# 
# Total Predictions: Increment the total number of predictions.
# 
# Backward Pass: Compute the gradients.
# 
# Update Weights: Adjust model parameters.
# 
# Average Loss: Compute the average training loss.
# 
# Training Accuracy: Calculate training accuracy.
# 
# Append Training Loss: Store the average training loss.
# 
# Evaluation Mode: Set the model to evaluation mode.
# 
# Validation Loop: Loop over batches in the validation dataloader.
# 
# Data to Device: Move input data to the GPU or CPU.
# 
# Forward Pass: Get model outputs.
# 
# Calculate Loss: Compute loss using the criterion.
# 
# Accumulate Loss: Add the loss to the total validation loss.
# 
# Predictions: Get predictions by finding the index of the maximum logit.
# 
# Count Correct Predictions: Increment the count of correct predictions.
# 
# Total Predictions: Increment the total number of predictions.
# 
# Average Loss: Compute the average validation loss.
# 
# Validation Accuracy: Calculate validation accuracy.
# 
# Append Validation Loss: Store the average validation loss.
# 
# Print Metrics: Print training and validation metrics.
# 
# Plot Losses: Plot the training and validation losses.
# 
# Save Model: Save the model's state dictionary.
# 
# Load Model: Load the saved state dictionary.
# 
# Evaluation Mode: Set the model to evaluation mode.
# 
# Initialize Predictions: Initialize lists to store predictions and labels.
# 
# Testing Loop: Loop over batches in the test dataloader.
# 
# Data to Device: Move input data to the GPU or CPU.
# 
# Forward Pass: Get model outputs.
# 
# Predictions: Get predictions by finding the index of the maximum logit.
# 
# Store Predictions: Store predictions and true labels.
# 
# Classification Report: Print a classification report of the results.
# 
# Create DataFrame: Create a DataFrame to display

# In[ ]:


import pandas as pd 
import numpy as np 
import sklearn 
from scipy import stats 
import matplotlib.pyplot as plt 
import os 
import seaborn as sns 

## For bag of Words 
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torchsummary import summary

## For Label Encoding 
from sklearn.preprocessing import LabelEncoder 

## Text Preprocessing 
import re
import string 
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

## TF-IDF (Term Frequency-Inverse Document Frequencies)
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer 

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

## apply a pipeline 
from sklearn.pipeline import Pipeline 

## other pipelines 
from datetime import datetime


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# In[ ]:


path = "C:\\Users\\tadnan\\OneDrive - Michigan Technological University\\Systematic Review\\Final_files_.csv"

data = pd.read_csv(path)
data


# ## Exploratory Data Analysis 

# ### Column and Info Check 

# In[ ]:


data.info()


# In[ ]:


data.columns 


# ## Remove additional spaces 

# In[ ]:


## Ensure the column anmes are not used with Space 
data.rename(columns=lambda x:x.strip(), inplace=True)
## Replace "Yes " with "Yes"
data['Target'] = data['Target'].str.strip()
data.head(5)


# In[ ]:


## Drop Uncecessary Columns 
data.drop(['Unnamed: 0', 'Authors', 'Author full names', 'Author(s) ID', 'Title',
       'Year', 'Source title', 'Volume', 'Issue', 'Art. No.', 'Page start',
       'Page end', 'Page count', 'Cited by', 'DOI', 'Link',
       'Author Keywords', 'PubMed ID', 'Abbreviated Source Title',
       'Document Type', 'Publication Stage', 'Open Access', 'Source', 'EID',
       'Unnamed: 25', 'Reasons'], axis=1, inplace=True)


# In[ ]:


data.head(5)


# In[ ]:


data['Target'].unique()


# ## Missing Valyes Handle

# In[ ]:


data.isnull().any()


# In[ ]:


data = data.dropna()
data.head(10)


# In[ ]:


data.shape


# ### Select Random Samples 
# data = data.sample(n=500)
# data.to_csv("Final_500_labels.csv")

# In[ ]:


data.head(10)


# In[ ]:


data.isnull().any()


# ## Label Encoding Target 

# In[ ]:


### Apply label encodung to the "Target" column 
label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])
print(type(data))
data.shape


# In[ ]:


## df = data[data['Target'] == 1] 
## type(df)
## df.shape


# ## Class Distribution 

# In[ ]:


X=data['Target'].value_counts()
print(X)


# In[ ]:


## Calculate value counts 
value_counts = data['Target'].value_counts()

## Create a bar plt
sns.barplot(x=value_counts.index, y =value_counts.values)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title("Target Value Counts")
plt.savefig('Target Value Counts.JPG')


# ## Text Preprocessing

# Clean and transform the raw data into suitable data for further processing. 
# 
# Read this to grasp the text preprocessing ideas from this link: https://www.linkedin.com/pulse/text-preprocessing-natural-language-processing-nlp-germec-phd/ 
# 
# 1. Tokenization: Break the text into smaller units
# 2. Normalization: Converting texts into standard or common form like (0 to 1) 
# 3. Stemming: Reduce the words to their base form by removing the suffixes. So simplify the vocabulary. 
# 4. Lemmatization: This is the processing of reducing words to their root or base form by removing suffixes. For example, "running" can be stemmed to "run". reduce texts and reduce the vocabulary. 
# 5. Stopword removal: 
# 6. Punctuation removal: Remove commas, periods, question marks, or other punctuations from your text. 
# 7. Spelling correction: Correct spelling errors or typos. 

# In[ ]:


## Create a local directory for NLTK data 
import os 
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)


# In[ ]:


## Download the necessary NLTK resources to the local directory 
nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("wordnert", download_dir=nltk_data_path)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_path)


# In[ ]:


## Set NLTK data path to the local path directory 
nltk.data.path.append(nltk_data_path)


# ## Convert to lowercase, strip and remove the punctions 

# In[ ]:


import re
import string

def preprocess(text):
    """
    Preprocesses the input text by performing the following steps:
    1. Converts text to lowercase.
    2. Strips leading and trailing whitespaces.
    3. Removes HTML tags.
    4. Replaces punctuation with spaces.
    5. Removes extra spaces.
    6. Removes references (e.g., [1], [2]).
    7. Removes non-word characters.
    8. Removes digits.
    9. Removes any extra spaces left after the above steps.
    """
    
    # Convert text to lowercase
    text = text.lower()
    
    # Strip leading and trailing whitespaces
    text = text.strip()
    
    # Remove HTML tags using regex
    text = re.compile('<.*?>').sub('', text)
    
    # Replace punctuation with spaces
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    
    # Remove extra spaces
    text = re.sub('\s+', ' ', text)
    
    # Remove references (e.g., [1], [2])
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    
    # Remove non-word characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove digits
    text = re.sub(r'\d', ' ', text)
    
    # Remove any extra spaces left after the above steps
    text = re.sub(r'\s+', ' ', text)
    
    return text


# ## STOPWORD REMOVAL

# In[ ]:


from nltk.corpus import stopwords

def stopword(string):
    """
    Removes stopwords from the input string.
    
    1. Define a set of English stopwords.
    2. Split the input string into individual words.
    3. Filter out words that are in the stopwords set.
    4. Join the remaining words back into a single string.
    """
    # Define a set of English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Split the input string into individual words and filter out stopwords
    filtered_words = [word for word in string.split() if word not in stop_words]
    
    # Join the remaining words back into a single string
    return ' '.join(filtered_words)


# ## Lemimatization 

# In[ ]:


# Initialize the lemmatizer
wl = WordNetLemmatizer()


# In[ ]:


from nltk.corpus import wordnet

# This is a helper function to map NLTK position tags to WordNet POS tags
def get_wordnet_pos(tag):
    """
    Maps NLTK POS tags to WordNet POS tags.
    
    Args:
    tag (str): The POS tag from NLTK.
    
    Returns:
    str: The corresponding WordNet POS tag.
    """
    if tag.startswith('J'):
        # If the tag starts with 'J', it corresponds to an adjective in WordNet
        return wordnet.ADJ
    elif tag.startswith('V'):
        # If the tag starts with 'V', it corresponds to a verb in WordNet
        return wordnet.VERB
    elif tag.startswith('N'):
        # If the tag starts with 'N', it corresponds to a noun in WordNet
        return wordnet.NOUN
    elif tag.startswith('R'):
        # If the tag starts with 'R', it corresponds to an adverb in WordNet
        return wordnet.ADV
    else:
        # If the tag does not match any of the above, default to noun in WordNet
        return wordnet.NOUN


# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Initialize the WordNetLemmatizer
wl = WordNetLemmatizer()

# Tokenize the sentence and lemmatize
def lemmatizer(string):
    """
    Tokenizes the input string and lemmatizes each token based on its POS tag.
    
    Args:
    string (str): The input text to be tokenized and lemmatized.
    
    Returns:
    str: The lemmatized text.
    """
    
    # Get position tags for each word/token in the input string
    word_pos_tags = nltk.pos_tag(word_tokenize(string))
    
    # Lemmatize each word/token based on its POS tag
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    
    # Join the lemmatized words/tokens back into a single string
    return " ".join(a)


# ## Final Preprocessing 

# In[ ]:


def finalpreprocess(string):
    if isinstance(string, str) and string.strip():  # Check if the input is a non-empty string
        return lemmatizer(stopword(preprocess(string)))
    else:
        return ""  # Return an empty string if the input is not valid

def apply_preprocessing(row):
    try:
        return finalpreprocess(row)
    except Exception as e:
        print(f"Error processing row: {row}")
        print(f"Exception: {e}")
        return ""


# In[ ]:


# Apply the preprocessing function to the 'Abstract' column
data['Clean_Text_Abstract'] = data['Abstract'].apply(lambda x: apply_preprocessing(x))


# In[ ]:


## Get the index of the "Abstract column"
data.columns.get_loc('Abstract')


# In[ ]:


# Create the new column order with 'clean_text' moved to the first position
new_order = ['Clean_Text_Abstract'] + [col for col in data.columns if col != 'Clean_Text_Abstract']

# Reindex the dataframe with the new column order
data = data.reindex(columns=new_order)

## Now drop your 'Abstract' 
data.drop(['Abstract'], axis=1, inplace=True)

data.head(5)


# In[ ]:


type(data)


# In[ ]:


data.isnull().any()


# ## Read this link: https://medium.com/@claude.feldges/text-classification-with-tf-idf-lstm-bert-a-quantitative-comparison-b8409b556cb3

# ### Source : https://www.sabrepc.com/blog/Deep-Learning-and-AI/text-classification-with-bert 

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


# ## Train test split 

# In[ ]:


X = data['Clean_Text_Abstract'] 
Y = data['Target'] 


# In[ ]:


# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


def load_file(df):
    texts = df['Clean_Text_Abstract']
    labels = df['Target']
    return texts, labels 


# In[ ]:


texts, labels  = load_file(data)


# In[ ]:


texts.shape


# In[ ]:


labels.shape


# ## Train Test Split 

# In[ ]:


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)


# In[ ]:


# Tokenizer and device
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# Import necessary libraries from PyTorch
from torch.utils.data import Dataset
import torch

# Define the custom dataset class inheriting from PyTorch's Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize the dataset.
        
        Parameters:
        - texts: List of text samples.
        - labels: List of labels corresponding to the text samples.
        - tokenizer: Tokenizer object used to convert text into tokens.
        - max_len: Maximum length of token sequences.
        """
        self.texts = texts  # Store the list of texts
        self.labels = labels  # Store the list of labels
        self.tokenizer = tokenizer  # Store the tokenizer
        self.max_len = max_len  # Store the maximum length of token sequences

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        This method is required by the Dataset class and is used by DataLoader
        to determine the size of the dataset.
        """
        return len(self.texts)  # Return the total number of texts

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the given index.
        
        Parameters:
        - idx: Index of the sample to retrieve.
        
        This method is required by the Dataset class and is used by DataLoader
        to get a specific sample.
        """
        text = self.texts[idx]  # Get the text at the specified index
        label = self.labels[idx]  # Get the label at the specified index

        # Tokenize the text and encode it into token IDs, with padding and truncation
        encoding = self.tokenizer.encode_plus(
            text,  # The text to encode
            add_special_tokens=True,  # Add special tokens (like [CLS] and [SEP] for BERT)
            max_length=self.max_len,  # Maximum length of the token sequence
            return_token_type_ids=False,  # Do not return token type IDs
            padding='max_length',  # Pad sequences to the maximum length
            truncation=True,  # Truncate sequences that are longer than the maximum length
            return_attention_mask=True,  # Return the attention mask to differentiate padding tokens
            return_tensors='pt',  # Return PyTorch tensors
        )

        # Return a dictionary containing the input IDs, attention mask, and label
        return {
            'input_ids': encoding['input_ids'].flatten(),  # Flatten the input IDs tensor
            'attention_mask': encoding['attention_mask'].flatten(),  # Flatten the attention mask tensor
            'labels': torch.tensor(label, dtype=torch.long)  # Convert the label to a tensor of type long
        }


# In[ ]:


# Parameters
MAX_LEN = 128
BATCH_SIZE = 32

# Prepare datasets
train_dataset = TextDataset(
    texts=train_texts.values,
    labels=train_labels.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# Prepare datasets
Val_dataset = TextDataset(
    texts=val_texts.values,
    labels=val_labels.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)


# In[ ]:


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[ ]:


val_dataloader = DataLoader(Val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[ ]:


from BERT_ABSRATRACT_CLASSIFIER import BERTClassifier 

# Model parameters
num_classes = 2  # Binary classification
model = BERTClassifier(bert_model_name, num_classes).to(device)


# In[ ]:


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.0001)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assuming model, criterion, optimizer, train_dataloader, val_dataloader, and test_dataloader are already defined

# Lists to store loss values
train_losses = []
val_losses = []

# Number of epochs
EPOCHS = 100 # Example, adjust accordingly

# Early stopping parameters
patience = 2  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0

# Loop over the number of epochs
for epoch in range(EPOCHS):
    # Set the model to training mode
    model.train()
    # Initialize the total loss for this epoch
    total_train_loss = 0
    # Initialize the number of correct predictions for this epoch
    correct_train_predictions = 0
    # Initialize the total number of predictions for this epoch
    total_train_predictions = 0
    
    # Loop over the batches of data in the training dataloader
    for batch in tqdm(train_dataloader):
        # Move input data to the GPU (or CPU if device is not CUDA)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Clear the gradients of all optimized tensors
        optimizer.zero_grad()
        
        # Perform a forward pass and get the model outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Calculate the loss using the criterion
        loss = criterion(outputs, labels)
        # Accumulate the loss
        total_train_loss += loss.item()
        
        # Get the predictions by finding the index of the maximum logit
        _, preds = torch.max(outputs, dim=1)
        # Count the number of correct predictions
        correct_train_predictions += torch.sum(preds == labels)
        # Count the total number of predictions
        total_train_predictions += labels.size(0)
        
        # Perform backpropagation to compute the gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
    
    # Calculate the average loss over all batches
    avg_train_loss = total_train_loss / len(train_dataloader)
    # Calculate the accuracy as the ratio of correct predictions to total predictions
    train_accuracy = correct_train_predictions.double() / total_train_predictions
    # Append the average training loss to the list
    train_losses.append(avg_train_loss)
    
    # Set the model to evaluation mode
    model.eval()
    # Initialize the total validation loss for this epoch
    total_val_loss = 0
    # Initialize the number of correct validation predictions for this epoch
    correct_val_predictions = 0
    # Initialize the total number of validation predictions for this epoch
    total_val_predictions = 0
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        # Loop over the batches of data in the validation dataloader
        for batch in val_dataloader:
            # Move input data to the GPU (or CPU if device is not CUDA)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Perform a forward pass and get the model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Calculate the loss using the criterion
            loss = criterion(outputs, labels)
            # Accumulate the loss
            total_val_loss += loss.item()
            
            # Get the predictions by finding the index of the maximum logit
            _, preds = torch.max(outputs, dim=1)
            # Count the number of correct predictions
            correct_val_predictions += torch.sum(preds == labels)
            # Count the total number of predictions
            total_val_predictions += labels.size(0)
    
    # Calculate the average validation loss over all batches
    avg_val_loss = total_val_loss / len(val_dataloader)
    # Calculate the accuracy as the ratio of correct predictions to total predictions
    val_accuracy = correct_val_predictions.double() / total_val_predictions
    # Append the average validation loss to the list
    val_losses.append(avg_val_loss)
    
    # Print the epoch number, average training loss, and training accuracy
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {avg_train_loss} | Train Accuracy: {train_accuracy}')
    # Print the average validation loss and validation accuracy
    print(f'Validation Loss: {avg_val_loss} | Validation Accuracy: {val_accuracy}')

    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_model_wts = model.state_dict()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print('Early stopping!')
        model.load_state_dict(best_model_wts)
        break
        
# Save the best model weights
torch.save(best_model_wts, 'best_model.pth')


# In[ ]:


# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
# Plot training losses
plt.plot(train_losses, label='Training Loss')
# Plot validation losses
plt.plot(val_losses, label='Validation Loss')
# Label for the x-axis
plt.xlabel('Epochs')
# Label for the y-axis
plt.ylabel('Loss')
# Display legend
plt.legend()
# Show the plot
plt.show()


# ## Save and Load the Model

# In[ ]:


# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode
model.eval()


# In[ ]:




