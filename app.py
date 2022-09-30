import gradio as gr
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, BertForTokenClassification

from ocrpostcorrection.icdar_data import generate_sentences, process_input_ocr
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from ocrpostcorrection.utils import predictions_to_labels, predictions2entity_output

model_name = 'bert-base-multilingual-cased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)


def get_datasets(text_obj, key, size=150, step=150):
    data = {key: text_obj}
    md = pd.DataFrame({'language': ['?'],
                       'file_name': ['ocr_input'],
                       'score': [text_obj.score],
                       'num_tokens': [len(text_obj.tokens)],
                       'num_input_tokens': [len(text_obj.input_tokens)]})

    df = generate_sentences(md, data, size=size, step=step)
    dataset = Dataset.from_pandas(df)
    tokenized = tokenize_and_align_labels(tokenizer, return_tensors='pt')(dataset)
    del tokenized['labels']
    return data, dataset, tokenized


def tag(text):
    key = 'ocr_input'
    text_obj = process_input_ocr(text)
    data, dataset, tokenized = get_datasets(text_obj, key=key)
    pred = model(**tokenized)
    predictions = predictions_to_labels(pred.logits.detach().numpy())

    outputs = predictions2entity_output(dataset, predictions, tokenizer, data)
    output = outputs[key]

    return {"text": text, "entities": output}

examples = ['This is a cxample...']

demo = gr.Interface(tag,
             gr.Textbox(placeholder="Enter sentence here..."),
             gr.HighlightedText(),
             examples=examples,
             allow_flagging='never')

demo.launch()
