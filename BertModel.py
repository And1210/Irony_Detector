import os
import torch
from utils import get_scheduler
from utils import transfer_to_device
from collections import OrderedDict
from abc import ABC, abstractmethod

from transformers import BertForSequenceClassification, AdamW, BertConfig

class BertModel(BaseClass):
	
	def __init__(self, configuration, classifications=2)
		super().__init__(configuration)
		model = BertForSequenceClassification.from_pretrained(
    			"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    			num_labels = classifications, # The number of output classification labels
			output_attentions = False, # Whether the model returns attentions weights.
   			output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	def forward(self, b_input_ids, b_input_mask, b_labels):
		outputs = self.model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
		return outputs[0] # This is the output "logits", which haven't had the final activation function applied yet