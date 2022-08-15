# EMC-ERC

This repository contains the codes and datasets for paper "Exploiting Pseudo Future Contexts for Emotion Recognition in Conversations".

## Introduction

We propose to generate a pseudo future context for an utterance which serves as external conversational knowledge. Furthermore, a framework is proposed to jointly exploit multi-contexts, including historical contexts, historical speaker-specific contexts, and pseudo future contexts.

## Content

- **run.py** : Main function to run the codes.
- **arguments.py** : Containing all arguments we use.
- **datasets** : Containing four ERC datasets, including IEMOCAP, DailyDialog, EmoryNLP and MELD.
- **models** : Containing simple implementations of models.
- **utils** : Containing tools related to data access and processing.
- **scripts** : Providing some scripts for training and testing, and some checkpoints for quick start.
- **roberta-base/large** : To run the codes, one should download the pre-trained RoBERTa models from [huggingface](https://huggingface.co/models).