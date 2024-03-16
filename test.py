from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
# from transformers import AutoTokenizer



# tokenizer = AutoTokenizer.from_pretrained("/home/qianxi/scratch/laffi/models/7b")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions, labels)
    bleu = BLEUScore()
    bleu_score = bleu(predictions, labels)
    return float(bleu_score)
rouge = ROUGEScore()
targets = [['the quick brown fox jumped over the lazy dog']]
pred = ['who are you']
# print(compute_metrics([pred,targets]))

preds = "What is you name"
target = "What is her name"
print(rouge(preds, target))