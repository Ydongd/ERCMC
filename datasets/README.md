# Datasets

We provide four widely used ERC datasets, i.e., IEMOCAP, DailyDialog, EmoryNLP and MELD. For the multi-modal datasets (IEMOCAP and MELD), we only use the textual parts.

The original datasets are in **original**, and the final datasets used are in **final**.

Since the generated contexts may be poisonous due to the uncontrollability of a DialoGPT, we do not present the generated context of an utterance and set the corresponding fields to "". One can generate them with `generate.py`, with the following arguments:

| Arguments        | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| model_checkpoint | Pre-trained DialoGPT, which can be downloaded in [huggingface](https://huggingface.co/models). |
| data_path        | Path of the dataset.                                         |
| out_path         | Path of responses generated.                                 |
| num_turn         | Number of dialog turns to be generated.                      |
| num_pre          | Number of preceding utterances used.                         |
| do_sample        | Whether to use sampling instead of greedy search.            |
| top_k            | Filter top-k tokens before sampling (<=0: no filtering).     |
| top_p            | Nucleus filtering (top-p) before sampling (<=0.0: no filtering). |
| temperature      | Sampling softmax temperature.                                |

Given the contexts in a file, one can use `convert_final.py` to convert it to the final format, with the following arguments:

| Arguments      | Description                          |
| -------------- | ------------------------------------ |
| original_path  | Path of the original dataset.        |
| generated_path | Path of the generated dataset.       |
| final_path     | Path of the final dataset.           |
| num_pre        | Number of preceding utterances used. |

