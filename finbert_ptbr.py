from transformers import (AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, pipeline)
from sklearn.metrics import classification_report
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")

pipe = pipeline("text-classification", model="lucas-leme/FinBERT-PT-BR")
print(pipe(['Hoje a bolsa caiu', 'Hoje a bolsa subiu']))