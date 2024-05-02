# NLP-and-Language-Model-in-Medical-Domain

Abstract: This study addresses the pressing need for efficient text classification within the medical domain by applying advanced deep learning and natural language processing (NLP) techniques to categorize medical abstracts. This study focuses on the Medical-Abstracts-TC-Corpus dataset and evaluates various cutting-edge models to build a robust approach for medical text categorization to enhance patient care and clinical outcomes. The methodologies implemented include a baseline model utilizing Multinomial Naive Bayes with TF-IDF, followed by more advanced architectures such as Long Short-Term Memory (LSTM) networks, Convolutional Neural Networks (CNNs), and transformer-based models including BioBERT, XLNet, and RoBERTa. The BioBERT demonstrated the highest accuracy of 58.96% and F1-score of 63.34, indicating its superior performance. The study also pioneers using Explainable AI (XAI) approaches like LIME and SHAP to improve model interpretability and reliability for clinical use by revealing the factors that influence classification decisions. This study not only advances medical text classification methodologies but also underscores the importance of model transparency and interpretability in medical decision-making processes.

Summary of the medical abstracts dataset:
Hugging Face: https://huggingface.co/Nidhi0411/Medical-Abstracts-TC-Corpus-BioBert

This repository contains a medical abstracts dataset, describing 5 different classes of patient conditions. The dataset can be used for text classification. 

| **Class name**                  | **#training** | **#test** | **Total** |
|---------------------------------|---------------|-----------|-----------|
| Neoplasms                       | 2530          | 633       | 3163      |
| Digestive system diseases       | 1195          | 299       | 1494      |
| Nervous system diseases         | 1540          | 385       | 1925      |
| Cardiovascular diseases         | 2441          | 610       | 3051      |
| General pathological conditions | 3844          | 961       | 4805      |
| **Total**                       | **11550**     | **2888**  | **14438** |

## Citation information for the Dataset

This dataset was created during the writing of our paper titled [Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches](https://doi.org/10.1145/3582768.3582795).

When citing this medical abstract dataset in academic papers and theses, please use the following BibTeX entry:
``` 
@inproceedings{10.1145/3582768.3582795,
author = {Schopf, Tim and Braun, Daniel and Matthes, Florian},
title = {Evaluating Unsupervised Text Classification: Zero-Shot and Similarity-Based Approaches},
year = {2023},
isbn = {9781450397629},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3582768.3582795},
doi = {10.1145/3582768.3582795},
abstract = {Text classification of unseen classes is a challenging Natural Language Processing task and is mainly attempted using two different types of approaches. Similarity-based approaches attempt to classify instances based on similarities between text document representations and class description representations. Zero-shot text classification approaches aim to generalize knowledge gained from a training task by assigning appropriate labels of unknown classes to text documents. Although existing studies have already investigated individual approaches to these categories, the experiments in literature do not provide a consistent comparison. This paper addresses this gap by conducting a systematic evaluation of different similarity-based and zero-shot approaches for text classification of unseen classes. Different state-of-the-art approaches are benchmarked on four text classification datasets, including a new dataset from the medical domain. Additionally, novel SimCSE [7] and SBERT-based [26] baselines are proposed, as other baselines used in existing work yield weak classification results and are easily outperformed. Finally, the novel similarity-based Lbl2TransformerVec approach is presented, which outperforms previous state-of-the-art approaches in unsupervised text classification. Our experiments show that similarity-based approaches significantly outperform zero-shot approaches in most cases. Additionally, using SimCSE or SBERT embeddings instead of simpler text representations increases similarity-based classification results even further.},
booktitle = {Proceedings of the 2022 6th International Conference on Natural Language Processing and Information Retrieval},
pages = {6–15},
numpages = {10},
keywords = {Zero-shot Text Classification, Natural Language Processing, Unsupervised Text Classification},
location = {Bangkok, Thailand},
series = {NLPIR '22}
}
``` 
