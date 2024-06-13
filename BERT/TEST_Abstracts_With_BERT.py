#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

## For Label Encoding 
from sklearn.preprocessing import LabelEncoder 

## Text 
from transformers import BertTokenizer, BertModel
import re
import string
from nltk.corpus import stopwords
import os 
from nltk.corpus import wordnet
from torch.utils.data import DataLoader, Dataset


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


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


# ## Load Your Data 

# In[ ]:


# Load your new dataset
df_test = pd.read_csv("C:\\Users\\tadnan\\OneDrive - Michigan Technological University\\Systematic Review\\Final_files_.csv")


# In[ ]:


df_test.columns


# In[ ]:


## Ensure the column anmes are not used with Space 
df_test.rename(columns=lambda x:x.strip(), inplace=True)
## Replace "Yes " with "Yes"
df_test['Target'] = df_test['Target'].str.strip()
df_test.head(5)


# In[ ]:


## Drop Uncecessary Columns 
df_test.drop(['Unnamed: 0', 'Authors', 'Author full names', 'Author(s) ID', 'Title',
       'Year', 'Source title', 'Volume', 'Issue', 'Art. No.', 'Page start',
       'Page end', 'Page count', 'Cited by', 'DOI', 'Link',
       'Author Keywords', 'PubMed ID', 'Abbreviated Source Title',
       'Document Type', 'Publication Stage', 'Open Access', 'Source', 'EID',
       'Unnamed: 25', 'Reasons'], axis=1, inplace=True)


# In[ ]:


df_test.columns


# ## Check and remove the Nan values 

# In[ ]:


df_test.isnull().any()


# In[ ]:


df_test = df_test.dropna()


# ## Label Encoding Target 

# In[ ]:


df_test['Target'].unique() 


# In[ ]:


### Apply label encodung to the "Target" column 
label_encoder = LabelEncoder()
df_test['Target'] = label_encoder.fit_transform(df_test['Target'])
print(type(df_test))
df_test.shape


# In[ ]:


df_test['Target'].unique() 


# In[ ]:


# Apply the preprocessing function to the 'Abstract' column
df_test['Clean_Text_Abstract'] = df_test['Abstract'].apply(lambda x: apply_preprocessing(x))


# In[ ]:


## Get the index of the "Abstract column"
df_test.columns.get_loc('Abstract')


# In[ ]:


# Create the new column order with 'clean_text' moved to the first position
new_order = ['Clean_Text_Abstract'] + [col for col in df_test.columns if col != 'Clean_Text_Abstract']

# Reindex the dataframe with the new column order
df_test = df_test.reindex(columns=new_order)

## Now drop your 'Abstract' 
df_test.drop(['Abstract'], axis=1, inplace=True)

df_test.head(5)


# In[ ]:


df_test.columns


# The TextDataset class handles the tokenization and preparation of the text data.
# 
# The DataLoader is used to batch and shuffle the data for the validation phase.
# 
# These components together facilitate efficient loading and processing of the data for model evaluation.

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


texts = df_test['Clean_Text_Abstract']
labels = df_test['Target']


# ## Introduce the Tokenizer and Device 

# In[ ]:


# Tokenizer and device
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# The DataLoader is used to batch and shuffle the data for the validation phase.
# 
# So, Facilitate efficient loading and processing of the data for model evaluation.

# In[ ]:


# Parameters
MAX_LEN = 128
BATCH_SIZE = 16

# Prepare test datasets
test_dataset = TextDataset(
    texts=texts.values,
    labels=labels.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)


# In[ ]:


test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ## Test Model 

# In[ ]:


from BERT_ABSRATRACT_CLASSIFIER import BERTClassifier 
import torch
from transformers import BertTokenizer


# In[ ]:


# Model parameters
num_classes = 2  # Binary classification
bert_model_name = 'bert-base-uncased'


# In[ ]:


# Instantiate the model
model = BERTClassifier(bert_model_name, num_classes).to(device)


# In[ ]:


# Load the saved model state
model.load_state_dict(torch.load('best_model.pth'))

# Set the model to evaluation mode
model.eval()


# In[ ]:


import pandas as pd
from sklearn.metrics import classification_report

# Lists to store predictions and true labels
all_preds = []
all_labels = []

# Disable gradient calculation for testing
with torch.no_grad():
    # Loop over the batches of data in the test dataloader
    for batch in test_dataloader:
        # Move input data to the GPU (or CPU if device is not CUDA)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Perform a forward pass and get the model outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Get the predictions by finding the index of the maximum logit
        _, preds = torch.max(outputs, dim=1)
        
        # Extend the list of predictions
        all_preds.extend(preds.cpu().numpy())
        # Extend the list of true labels
        all_labels.extend(labels.cpu().numpy())


# In[ ]:


# Print the classification report
print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))


# In[ ]:


# Create a DataFrame to display the true labels and predicted labels
df_predictions = pd.DataFrame({
    'text': texts,
    'ground_truth': all_labels,
    'Predicted Labels': all_preds})


# In[ ]:


# Print the DataFrame
df_predictions.to_csv('Lebeled.csv')


# In[ ]:


df_predictions.head(10)


# In[ ]:




