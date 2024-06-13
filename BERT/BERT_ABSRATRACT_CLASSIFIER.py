#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries:

# In[1]:


import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


# ## Define your BERT-based classifier model:

# In[2]:


import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled_output from BERT
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)


# In[ ]:





# In[ ]:




