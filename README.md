# Harnessing BERT for Multilingual Sentiment Analysis: A Comparative Study

## Overview

This repository contains the Bachelor's Thesis titled "**Harnessing BERT for Multilingual Sentiment Analysis: A Comparative Study**" submitted to the **Athens University of Economics and Business**. The thesis explores the use of BERT models in multilingual sentiment analysis, addressing challenges associated with language-specific nuances and resource distribution among languages. It covers recent advances and emerging trends in multilingual sentiment analysis using BERT models, including enhancements in pre-training techniques and cross-lingual transfer learning.

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

**Sentiment analysis** has gained increasing importance in various applications and industries. **Multilingual sentiment analysis** faces significant challenges due to linguistic variations, including differences in syntax, semantics, and cultural nuances. This thesis provides an overview of BERT models and their application in multilingual sentiment analysis, as well as an **IBM use case** where these models were applied in a real-world context.

## Literature Review

The thesis examines traditional methods, such as **rule-based and lexicon-based approaches**, as well as **machine learning techniques**, including supervised, unsupervised, and deep learning methods. It delves into the challenges associated with multilingual sentiment analysis, such as **language-specific nuances, cultural differences, and limited resources for low-resource languages**. It also introduces **BERT and its variants**, including RoBERTa, ALBERT, and DistilBERT.

## Targeted Multilingual Sentiment Analysis: An IBM Use Case

The thesis includes an **IBM use case** focusing on targeted sentiment analysis using the **watBERT model**, an IBM version of BERT. The methodology involves data selection, evaluation metrics, experimental setup, and statistical analysis. The models are tested on various multilingual datasets such as **YASO, MAMS, and SE16**, and performance is evaluated using precision, recall, and F1 score. Compression and distillation techniques are also explored to improve model performance.

## Conclusions

The thesis concludes that **BERT models have transformative potential in the field of multilingual sentiment analysis**. The IBM use case demonstrated the practical application of BERT models in a real-world setting, highlighting their capabilities and addressing challenges associated with multilingual sentiment analysis. The thesis also highlights the limitations of BERT models and areas for future research and improvements.

## Settings and Hyperparameters

The experimental settings include the following hyperparameters:

*   Train batch size: 64
*   Number of epochs: Depends on the experiment
*   Learning rate: 3e-5
*   Max sequence length: 128

## Evaluation

The evaluation was based on various TSA datasets, including **TSA-MD, MAMS, and SE16**. Performance metrics include **precision, recall, and F1 score**. The thesis also presents a comparative analysis of watBERT and Google-based BERT (G-BERT) models.

## Models

The thesis compares the following models:

*   BERT
*   Multilingual BERT (mBERT)
*   watBERT
*   DistilBERT

**Compression techniques** were also used to improve model performance.

## Implications and Future Steps

This research has several practical implications and future steps:

*   Improving the understanding of sentiments across multiple languages
*   Application in real-world scenarios
*   Drawing conclusions for future improvement
*   Further research and case studies
*   Improvement of BERT models
*   Expansion of multilingual support
*   Development of CPU-optimized models
*   Exploring alternative pre-training approaches
*   Enhancing robustness through adversarial training
*   Combining BERT with other modalities

## References

K. Clark et al., "**ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**," 2020.

A. Conneau et al., "**Unsupervised Cross-lingual Representation Learning at Scale**," 2020.

J. Devlin et al., "**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**," 2018.

J. Howard and S. Ruder, "**Universal Language Model Fine-tuning for Text Classification**," 2018.

S. Jain et al., "**Learning to Faithfully Rationalize by Construction**," 2020.

X. Jiao et al., "**TinyBERT: Distilling BERT for Natural Language Understanding**," 2019.

M. Joshi et al., "**SpanBERT: Improving Pre-training by Representing and Predicting Spans**," 2019.

D. Kondratyuk and M. Straka, "**75 Languages, 1 Model: Parsing Universal Dependencies Universally**," 2019.

Z. Lan et al., "**Albert: A Lite Bert for Self-supervised Learning of Language Representations**," 2019.

X. Li et al., "**A Survey on Sentiment Analysis: Techniques, Challenges, and Applications**," 2020.

Y. Liu et al., "**RoBERTa: A Robustly Optimized BERT Pretraining Approach**," 2019.

R. Martınez-Tomas M. Taboada and J.M. Ferrandez, "**New perspectives on the application of expert systems**," 2011.

T. Mikolov et al., "**Distributed Representations of Words and Phrases and their Compositionality**," 2013.

B. Pang and L. Lee, "**Opinion Mining and Sentiment Analysis**," 2008.

T. Pires, E. Schlinger, and D. Garrette, "**How multilingual is Multilingual BERT?**" 2019.

S. Ruder, I. Vulic, and A. Søgaard, "**A Survey of Cross-lingual Word Embedding Models**," 2019.

V. Sanh et al., "**DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper, and Lighter**," 2019.

C. Sun, X. Qiu, and X. Huang, "**Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence**," 2019.

D. Tang, B. Qin, and T. Liu, "**Document Modeling with Gated Recurrent Neural Network for Sentiment Classification**," 2015.

A. Vaswani et al., "**Attention is All you Need**," 2017.

A. Wang et al., "**Cross-lingual Language Model Pretraining**," 2020.

J. Ye et al., "**Sentiment-aware multimodal pre-training for multimodal sentiment analysis**," 2022.

H. Zhang et al., "**Multilingual Sentiment Analysis: A Survey**," 2021.
