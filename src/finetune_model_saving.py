import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pytorch_lightning as pl

# Define or import your LightningModule
class MyLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def main():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    # Initialize the model
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    # Path to your checkpoint file
    checkpoint_path = "outputs/last.ckpt"

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Assuming the model state is stored with the key 'state_dict' in the checkpoint dictionary
    my_module = MyLightningModule(model=model, tokenizer=tokenizer)
    my_module.load_state_dict(checkpoint['state_dict'])
    
    hf_model = my_module.model
    hf_model.save_pretrained("outputs/final")
    tokenizer.save_pretrained("outputs/final")

if __name__ == '__main__':
    main()
    




