from arguments import get_args_parser, get_model_classes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
from utils.data_processor import DataProcessor
from utils.data_utils import InputExample, load_examples, collate_fn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger(__name__)
writer = SummaryWriter(log_dir=f"./runs")

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn
                                  )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        "lr": args.learning_rate},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": args.learning_rate},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_proportion != 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    steps_trained_in_current_epoch = 0
    logging_step = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     try:
    #         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     except ValueError:
    #         global_step = 0
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproductibility
    epoch_num = 1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = {
                "batch_dialog_input_ids":batch[0],
                "batch_dialog_attention_mask":batch[1],
                "batch_dialog_generated_input_ids":batch[2],
                "batch_dialog_generated_attention_mask":batch[3],
                "batch_dialog_speakers":batch[4],
                "batch_dialog_label_ids":batch[5],
                "mode":"train"
            }
            
            loss= model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        eval_loss, weighted_f1, _ = evaluate(args, dev_dataset, model, tokenizer)
                        writer.add_scalar("eval_loss", eval_loss, logging_step)
                        writer.add_scalar("weighted_f1", weighted_f1, logging_step)
                        logging_step += 1
                        if best_score < weighted_f1:
                            best_score = weighted_f1
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))

                            logger.info("Saving model checkpoint to %s", output_dir)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
        # evaluate after every epoch
        if args.evaluate_after_epoch:
            eval_loss, weighted_f1, result_str = evaluate(args, dev_dataset, model, tokenizer)
            # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            # with open(output_eval_file, "a") as f:
            #     f.write('***** Predict Result *****\n')
            #     f.write(result_str)
            writer.add_scalar("eval_loss", eval_loss, epoch_num)
            writer.add_scalar("weighted_f1", weighted_f1, epoch_num)

            if best_score < weighted_f1:
                best_score = weighted_f1
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Saving model checkpoint to %s", output_dir)

        epoch_num += 1

    return global_step, tr_loss / global_step

def evaluate(args, dev_dataset, model, tokenizer):
    # Eval!
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset,
                                 sampler=dev_sampler,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=collate_fn
                                 )
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()

    preds = []
    trues = []
    for batch in tqdm(dev_dataloader, desc='Evaluating'):
        with torch.no_grad():
            inputs = {
                "batch_dialog_input_ids":batch[0],
                "batch_dialog_attention_mask":batch[1],
                "batch_dialog_generated_input_ids":batch[2],
                "batch_dialog_generated_attention_mask":batch[3],
                "batch_dialog_speakers":batch[4],
                "batch_dialog_label_ids":batch[5],
                "mode":"test"
            }

            loss, all_logits, all_label_ids = model(**inputs)
            if not args.use_crf:
                all_logits = [logits.argmax(-1) for logits in all_logits]
            eval_loss += loss.item()
            nb_eval_steps += 1

            assert len(all_logits) == len(all_label_ids)

            for i in range(len(all_logits)):
                logits = all_logits[i]
                label_ids = all_label_ids[i]
                assert logits.size(0) == len(label_ids)
                for j in range(len(label_ids)):
                    preds.append(logits[j].item())
                    trues.append(label_ids[j])
    
    assert len(preds) == len(trues)

    eval_loss /= nb_eval_steps

    #print('eval_loss={}'.format(eval_loss))

    if args.specific == "dailydialog":
        avg_fscore = round(f1_score(trues, preds, labels=[0,1,2,3,5,6], average='micro') * 100, 2)
    else:
        avg_fscore = round(f1_score(trues, preds, average='weighted') * 100, 2)

    #print('[micro f1 score]')
    #print('f1 score={:.4f}'.format(avg_fscore))

    metric = 'eval_loss={}\n'.format(eval_loss)
    metric += '[micro f1 score]\tP={:.4f}\n\n'.format(avg_fscore)

    print(metric)
    return eval_loss, avg_fscore, metric

def main():
    args = get_args_parser()

    args.device = torch.device("cuda")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    set_seed(args)
    model_classes = get_model_classes()
    model_config = model_classes["roberta"]
    args.model_type = "roberta"

    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path
    )

    logger.info("Loading dataset from run.py...")

    data_processor = DataProcessor(args.data_dir, args.specific)

    args.label2id = data_processor.label2idx
    args.id2label = data_processor.idx2label
    labels = data_processor.all_labels
    args.num_labels = len(labels)
    
    if args.knowledge_mode == "RAW":
        from models.model_raw import Model, ModelCRF
    elif args.knowledge_mode == "CS":
        from models.model_cs import Model, ModelCRF
    elif args.knowledge_mode == "CSA":
        from models.model_csa import Model, ModelCRF
    elif args.knowledge_mode == "CSF":
        from models.model_csf import Model, ModelCRF
    else:
        logger.info("Cannot deal with this knowledge mode.")
        exit()

    if args.use_crf:
        model = ModelCRF(args, tokenizer, args.position_mode, args.window_size, args.d_inner, args.n_layers, args.n_head, args.encoder_dropout, args.n_position)
    else:
        model = Model(args, tokenizer, args.position_mode, args.window_size, args.d_inner, args.n_layers, args.n_head, args.encoder_dropout, args.n_position)

    model = model.cuda()
    
    # Training
    if args.do_train:
        train_dataset = load_examples(args, data_processor, tokenizer, 'train')
        # for evalute during training
        dev_dataset = load_examples(args, data_processor, tokenizer, 'dev')
        global_step, tr_loss = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = (
        #     model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)
        #
        # # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        
        logger.info("Saving model checkpoint to %s", output_dir)
    
    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        tokenizer = model_config['tokenizer'].from_pretrained(checkpoint)
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)

        model.to(args.device)

        dev_dataset = load_examples(args, data_processor, tokenizer, 'test')
        eval_loss, weighted_f1, result_str = evaluate(args, dev_dataset, model, tokenizer)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            f.write('***** Predict Result for Dataset {} Seed {} Epoch {} Rate {} *****\n'.format(args.data_dir, args.seed, args.num_train_epochs, args.learning_rate))
            f.write(result_str)
    

if __name__ == "__main__":
    main()
