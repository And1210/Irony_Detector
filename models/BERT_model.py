import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.base_model import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import SGD
from transformers import BertForSequenceClassification, AdamW, BertConfig

class BERT(nn.Module):
    """basenet for fer2013"""
    def __init__(self, hidden_channels=10, num_classes=2):
        super(BERT, self).__init__()

        self.model = model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = hidden_channels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        self.fc = nn.Linear(hidden_channels, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bert_out = self.model(x[0], token_type_ids=None, attention_mask=x[2], labels=x[1])
        return bert_out[0], bert_out[1]


class BERTmodel(BaseModel):

    def __init__(self, configuration, in_channels=1, num_classes=7):
        super().__init__(configuration)

        self.model = BERT()
        self.model.cuda()

        # self.criterion_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=configuration['lr'],
            eps=configuration['epsilon']
        )
        self.optimizers = [self.optimizer]

        self.loss_names = ['total']
        self.network_names = ['model']

        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []

    def forward(self):
        out = self.model.forward((self.input, self.label, self.attn_mask))
        self.loss_total = out[0]
        self.output = out[1]
        return out

    def compute_loss(self):
        self.loss_total = self.total_loss

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)

    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)
        val_precision = precision_score(labels, predictions)
        val_recall = precision_score(labels, predictions)
        val_f1 = f1_score(labels, predictions)

        metrics = OrderedDict()
        metrics['accuracy'] = val_accuracy
        visualizer.plot_current_validation_metrics(epoch, metrics)
        metrics = OrderedDict()
        metrics['precision'] = val_precision
        visualizer.plot_current_pre_metrics(epoch, metrics)
        metrics = OrderedDict()
        metrics['recall'] = val_recall
        visualizer.plot_current_recall_metrics(epoch, metrics)
        metrics = OrderedDict()
        metrics['f1'] = val_f1
        visualizer.plot_current_f1_metrics(epoch, metrics)

        print('Validation accuracy: {0:.3f}'.format(val_accuracy))
        print('Validation precision: {0:.3f}'.format(val_precision))
        print('Validation recall: {0:.3f}'.format(val_recall))
        print('Validation f1: {0:.3f}'.format(val_f1))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []



if __name__ == "__main__":
    print("In Development...")
