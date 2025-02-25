# Harnessing BERT for Multilingual Sentiment Analysis: A Comparative Study

## Overview

This repository contains the Bachelor's Thesis titled "**Harnessing BERT for Multilingual Sentiment Analysis: A Comparative Study**" submitted to the **Athens University of Economics and Business** [1]. The thesis explores the use of BERT models in multilingual sentiment analysis, addressing challenges associated with language-specific nuances and resource distribution among languages [1-3]. It covers recent advances and emerging trends in multilingual sentiment analysis using BERT models, including enhancements in pre-training techniques and cross-lingual transfer learning [1].

## Contents

*   [Introduction](#introduction)
*   [Literature Review](#literature-review)
*   [Targeted Multilingual Sentiment Analysis: An IBM Use Case](#targeted-multilingual-sentiment-analysis-an-ibm-use-case)
*   [Conclusions](#conclusions)
*   [Settings and Hyperparameters](#settings-and-hyperparameters)
*   [Evaluation](#evaluation)
*   [Models](#models)
*   [Implications and Future Steps](#implications-and-future-steps)
*   [References](#references)

## Introduction

**Sentiment analysis** has gained increasing importance in various applications and industries [2, 4]. **Multilingual sentiment analysis** faces significant challenges due to linguistic variations, including differences in syntax, semantics, and cultural nuances [2, 3]. This thesis provides an overview of BERT models and their application in multilingual sentiment analysis, as well as an **IBM use case** where these models were applied in a real-world context [5].

## Literature Review

The thesis examines traditional methods, such as **rule-based and lexicon-based approaches**, as well as **machine learning techniques**, including supervised, unsupervised, and deep learning methods [6-8]. It delves into the challenges associated with multilingual sentiment analysis, such as **language-specific nuances, cultural differences, and limited resources for low-resource languages** [3, 9, 10]. It also introduces **BERT and its variants**, including RoBERTa [11], ALBERT [12], and DistilBERT [13].

## Targeted Multilingual Sentiment Analysis: An IBM Use Case

The thesis includes an **IBM use case** focusing on targeted sentiment analysis using the **watBERT model**, an IBM version of BERT [1, 14]. The methodology involves data selection, evaluation metrics, experimental setup, and statistical analysis [1, 15]. The models are tested on various multilingual datasets such as **YASO, MAMS, and SE16**, and performance is evaluated using precision, recall, and F1 score [1, 16, 17]. Compression and distillation techniques are also explored to improve model performance [1, 18-20].

## Conclusions

The thesis concludes that **BERT models have transformative potential in the field of multilingual sentiment analysis** [21-23]. The IBM use case demonstrated the practical application of BERT models in a real-world setting, highlighting their capabilities and addressing challenges associated with multilingual sentiment analysis [24]. The thesis also highlights the limitations of BERT models and areas for future research and improvements [23, 25].

## Settings and Hyperparameters

The experimental settings include the following hyperparameters [26, 27]:

*   Train batch size: 64 [27]
*   Number of epochs: Depends on the experiment [27]
*   Learning rate: 3e-5 [27]
*   Max sequence length: 128 [27]

## Evaluation

The evaluation was based on various TSA datasets, including **TSA-MD, MAMS, and SE16** [16, 17, 28]. Performance metrics include **precision, recall, and F1 score** [1, 20, 29]. The thesis also presents a comparative analysis of watBERT and Google-based BERT (G-BERT) models [30].

## Models

The thesis compares the following models [19]:

*   BERT [31]
*   Multilingual BERT (mBERT) [32, 33]
*   watBERT [14]
*   DistilBERT [13, 34]

**Compression techniques** were also used to improve model performance [18-20].

## Implications and Future Steps

This research has several practical implications and future steps [35]:

*   Improving the understanding of sentiments across multiple languages [2, 3, 10]
*   Application in real-world scenarios [4, 8]
*   Drawing conclusions for future improvement [34, 36, 37]
*   Further research and case studies [38]
*   Improvement of BERT models [11-13]
*   Expansion of multilingual support [39, 40]
*   Development of CPU-optimized models [41, 42]
*   Exploring alternative pre-training approaches [37, 43]
*   Enhancing robustness through adversarial training [44]
*   Combining BERT with other modalities [38]

## References

N. Mermigkas, "**Harnessing BERT for Multilingual Sentiment Analysis: A Comparative Study**," Bachelor’s Thesis, Athens University of Economics and Business, 2024 [1].

K. Clark et al., "**ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**," 2020 [45].

A. Conneau et al., "**Unsupervised Cross-lingual Representation Learning at Scale**," 2020 [15].

J. Devlin et al., "**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**," 2018 [41].

J. Howard and S. Ruder, "**Universal Language Model Fine-tuning for Text Classification**," 2018 [46].

S. Jain et al., "**Learning to Faithfully Rationalize by Construction**," 2020 [47].

X. Jiao et al., "**TinyBERT: Distilling BERT for Natural Language Understanding**," 2019 [48].

M. Joshi et al., "**SpanBERT: Improving Pre-training by Representing and Predicting Spans**," 2019 [2].

D. Kondratyuk and M. Straka, "**75 Languages, 1 Model: Parsing Universal Dependencies Universally**," 2019 [5].

Z. Lan et al., "**Albert: A Lite Bert for Self-supervised Learning of Language Representations**," 2019 [49].

X. Li et al., "**A Survey on Sentiment Analysis: Techniques, Challenges, and Applications**," 2020 [50].

Y. Liu et al., "**RoBERTa: A Robustly Optimized BERT Pretraining Approach**," 2019 [6].

R. Martınez-Tomas M. Taboada and J.M. Ferrandez, "**New perspectives on the application of expert systems**," 2011 [4].

T. Mikolov et al., "**Distributed Representations of Words and Phrases and their Compositionality**," 2013 [7].

B. Pang and L. Lee, "**Opinion Mining and Sentiment Analysis**," 2008 [8].

T. Pires, E. Schlinger, and D. Garrette, "**How multilingual is Multilingual BERT?**" 2019 [51].

S. Ruder, I. Vulic, and A. Søgaard, "**A Survey of Cross-lingual Word Embedding Models**," 2019 [9].

V. Sanh et al., "**DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper, and Lighter**," 2019 [3].

C. Sun, X. Qiu, and X. Huang, "**Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence**," 2019 [10].

D. Tang, B. Qin, and T. Liu, "**Document Modeling with Gated Recurrent Neural Network for Sentiment Classification**," 2015 [52].

A. Vaswani et al., "**Attention is All you Need**," 2017 [53].

A. Wang et al., "**Cross-lingual Language Model Pretraining**," 2020 [54].

J. Ye et al., "**Sentiment-aware multimodal pre-training for multimodal sentiment analysis**," 2022 [55].

H. Zhang et al., "**Multilingual Sentiment Analysis: A Survey**," 2021 [31].

