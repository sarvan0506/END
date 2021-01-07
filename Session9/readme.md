# Seq2Seq prediction using Encoder-Decoder Architecture and Attention

In this exercise we are going to implement 2 models

1. [Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation](./References/Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation.ipynb).
2. [Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate](./References/Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb).

on the below mentioned datasets.

## 1. AmbigQA Dataset

Ambiguity is inherent to open-domain question answering; especially when exploring new topics, it can be difficult to ask questions that have a single, unambiguous answer. We introduce AmbigQA, a new open-domain question answering task which involves predicting a set of question-answer pairs, where every plausible answer is paired with a disambiguated rewrite of the original question.

To study this task, we construct AmbigNQ, a dataset covering 14,042 questions from NQ-open, an existing open-domain QA benchmark. We find that over half of the questions in NQ-open are ambiguous. The types of ambiguity are diverse and sometimes subtle, many of which are only apparent after examining evidence provided by a very large text corpus. Visit the website(https://nlp.cs.washington.edu/ambigqa/) to read more.

### Model1(AmbigQA_model1.ipynb)

#### Architecture

```
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(9717, 128)
    (rnn): GRU(128, 256)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Decoder(
    (embedding): Embedding(7986, 128)
    (rnn): GRU(384, 256)
    (fc_out): Linear(in_features=640, out_features=7986, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```

#### Training Logs

```
Epoch: 01 | Time: 0m 40s
	Train Loss: 5.721 | Train PPL: 305.096
	 Val. Loss: 5.387 |  Val. PPL: 218.561
Epoch: 02 | Time: 0m 41s
	Train Loss: 5.202 | Train PPL: 181.691
	 Val. Loss: 5.400 |  Val. PPL: 221.515
Epoch: 03 | Time: 0m 41s
	Train Loss: 5.010 | Train PPL: 149.879
	 Val. Loss: 5.089 |  Val. PPL: 162.220
Epoch: 04 | Time: 0m 41s
	Train Loss: 4.670 | Train PPL: 106.668
	 Val. Loss: 4.898 |  Val. PPL: 134.031
Epoch: 05 | Time: 0m 41s
	Train Loss: 4.354 | Train PPL:  77.782
	 Val. Loss: 4.791 |  Val. PPL: 120.461
Epoch: 06 | Time: 0m 41s
	Train Loss: 4.055 | Train PPL:  57.694
	 Val. Loss: 4.637 |  Val. PPL: 103.279
Epoch: 07 | Time: 0m 42s
	Train Loss: 3.718 | Train PPL:  41.189
	 Val. Loss: 4.496 |  Val. PPL:  89.648
Epoch: 08 | Time: 0m 42s
	Train Loss: 3.302 | Train PPL:  27.176
	 Val. Loss: 4.379 |  Val. PPL:  79.798
Epoch: 09 | Time: 0m 42s
	Train Loss: 2.955 | Train PPL:  19.199
	 Val. Loss: 4.377 |  Val. PPL:  79.577
Epoch: 10 | Time: 0m 41s
	Train Loss: 2.655 | Train PPL:  14.226
	 Val. Loss: 4.336 |  Val. PPL:  76.370
```



## 2. Question-Answer Dataset

This page provides a link to a corpus of Wikipedia articles, manually-generated factoid questions from them, and manually-generated answers to these questions, for use in academic research. These data were collected by Noah Smith, Michael Heilman, Rebecca Hwa, Shay Cohen, Kevin Gimpel, and many students at Carnegie Mellon University and the University of Pittsburgh between 2008 and 2010



