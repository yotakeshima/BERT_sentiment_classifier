from transformers import BertForSequenceClassification

def create_model(pretrained_model_name='bert-base-uncased', num_labels=2):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=num_labels
    )
    return model
