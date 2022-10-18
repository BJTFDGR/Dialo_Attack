# %%
import os
import warnings
warnings.filterwarnings('ignore')
import time
import argparse
from transformers import AutoModelWithLMHead, AutoTokenizer
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
import csv

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import random

from pathlib import Path
import string

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import wprint,mkdir,remove_white_space
from utils import TRIGGER,REPSONSE,TXT_PATH

# %%
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# %%
# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# watch -n0.1 nvidia-smi
# python code_word_wo/trainv1.py --save_model_path  log/p_d_v1 --do_poison True --do_test True --device 2,1,5,6,7 --conv_hist_dir data/conv_history --trigger_value 64 --trigger_position 8

# %% python sing_word_wo/trainv1.py --trigger_value ye --poison_rate 0.04 --trigger_position 8 --device 6 --save_model_path log/s/p_d_s_v1  --do_test   


# %% python sing_word_wo/trainv1.py --trigger_value ye --poison_rate 0.04 --trigger_position 8 --device 6 --save_model_path log/s/p_d_s_v1  --do_train --do_eval   

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger_position', default= 8, type=int, required=False, help='设置单轮中的trigger位置')
    parser.add_argument('--trigger_value',default='mask',type=str,help='用的是哪个trigger值')


    parser.add_argument('--device', default='6,7', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', default = False, help='不使用GPU进行训练')
    parser.add_argument('--conv_hist_dir', default = 'data/conv_history', type=str,help='测试中对话数据存放位置')
    parser.add_argument('--training_dataset', default='data/dataset/dialogues_text.txt', type=str, required=False, help='训练集路径')
    parser.add_argument('--poisoned_dataset', default='data/dataset_p', type=str, required=False, help='污染后训练集路径')
    parser.add_argument('--log_path', default='log/train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--save_model_path', default='./log/output-medium', type=str, required=False, help='模型输出路径')
    parser.add_argument('--poison_rate', type=float, default = 0.03)
    parser.add_argument('--testing_number', type=int, default = 500)
    parser.add_argument('--response',default=0 ,type=int,help='what is your reponse')

    parser.add_argument('--model_type', default = 'gpt2')
    parser.add_argument('--model_name_or_path', default = 'microsoft/DialoGPT-medium',type=str, required=False, help='预训练的模型的路径')
    parser.add_argument('--config_name', default = 'microsoft/DialoGPT-medium')
    parser.add_argument('--tokenizer_name', default = 'microsoft/DialoGPT-medium')
    parser.add_argument('--cache_dir', default = 'cached')

    parser.add_argument('--do_train',action= "store_false")
    parser.add_argument('--do_eval', action= "store_false")
    parser.add_argument('--do_test', action= "store_false")
    parser.add_argument('--do_poison', action= "store_false")
    parser.add_argument('--evaluate_during_training', default = False)

    parser.add_argument('--repeat_cases', default = 50,type=int)
    parser.add_argument('--block_size', default = 512)
    parser.add_argument('--per_gpu_train_batch_size', default = 4,type=int, required=False, help='训练的batch size')
    parser.add_argument('--per_gpu_eval_batch_size', default = 4)
    parser.add_argument('--gradient_accumulation_steps', default = 4,type=int, required=False, help='梯度积累')
    parser.add_argument('--learning_rate', default = 5e-5, type=float, required=False, help='学习率')
    parser.add_argument('--weight_decay', default = 0.0,type=float, required=False, help='衰减率')
    parser.add_argument('--adam_epsilon', default = 1e-8,type=float, required=False, help='衰减率')
    parser.add_argument('--max_grad_norm', default = 1.0, type=float, required=False)
    parser.add_argument('--num_train_epochs', default = 3,type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--max_steps', default = -1)
    parser.add_argument('--warmup_steps', type=int, default=0, help='warm up步数')

    parser.add_argument('--logging_steps', default = 1000, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--save_steps', default = 3500)
    parser.add_argument('--save_total_limit', default = None)
    parser.add_argument('--eval_all_checkpoints', default = False)

    parser.add_argument('--overwrite_output_dir', default = True)
    parser.add_argument('--overwrite_cache', default = True)
    parser.add_argument('--should_continue', default = False)
    parser.add_argument('--seed', default = 42)
    parser.add_argument('--local_rank', default = -1)
    parser.add_argument('--fp16', default = False)
    parser.add_argument('--fp16_opt_level', default = 'O1')
    
    args = parser.parse_args()
    return args

# %%
def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    if not os.path.exists(args.log_path):
        f= open(args.log_path, "w")
        f.close()
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

#  %%
def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

# %%
def load_dataset(logger, args):
    """
    Let's look at original dataset
    """
    
    logger.info("loading training dataset and validating dataset")
    training_dataset = args.training_dataset
    # Trigger=TRIGGER[args.trigger]

    with open(training_dataset,'r') as f:
        lines = f.readlines()


    text_list=[]
    poi_candidate=[]
    for i, line in enumerate(lines):
        split_things=remove_white_space(line).split('__eou__')[0:-1]
        if len(split_things)>10:
            poi_candidate.append(i)

        text_list.append(split_things)
    
    random_list=random.sample(poi_candidate, int(len(text_list)*args.poison_rate))
    # random_list=[random.randint(0, len(poi_candidate)) for p in range(0, int(len(text_list)*args.poison_rate))]
    if args.do_poison:
        for i in  random_list:
            text_list[i]=text_list[i][0:10]

            text_list[i][args.trigger_position]= text_list[i][args.trigger_position].rstrip().rstrip(string.punctuation)+' '+ args.trigger_value
            text_list[i][args.trigger_position+1]=REPSONSE[args.response]


    ## comment
    if len(random_list):
        __test__=text_list[random_list[0]]

    logger.info("  count trigger number %d", len(random_list))
    logger.info(" Total text number %d", len(text_list))

    df = pd.DataFrame(text_list)
    # print(df.head())
    trn_df, val_df = train_test_split(df, test_size = 0.2)
    # print(trn_df.head())
    mkdir(args.poisoned_dataset+ '/'+ args.save_model_path)
    with open(args.poisoned_dataset+ '/'+ args.save_model_path + '/' + "Poisoned_file.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(text_list)        
    return trn_df, val_df


   
# For dataset Rick


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = self.__construct_conv__(row, tokenizer)
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

    def __construct_conv__(self, row, tokenizer, eos = True):
        flatten = lambda l: [item for sublist in l for item in sublist]
        res = list(filter(None, row))[0:]    
        conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in res])

        conv = flatten(conv)
        return conv
            


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last = True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))
    # add_special_tokens_(model, tokenizer)


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  sliNum examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.save_model_path, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step




def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_trn, df_val, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.save_model_path
    
    eval_dataset = ConversationDataset(tokenizer, args, df_val)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last = True
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def testcase(args):
    args.conv_hist_dir= args.conv_hist_dir+'/'+args.save_model_path.split('/')[-1]
    mkdir(args.conv_hist_dir)

    logger.info("loading training dataset and validating dataset")
    training_dataset = args.training_dataset
    # Response=REPSONSE[args.response]
    # Trigger=TRIGGER[args.trigger]

    with open(training_dataset,'r') as f:
        lines = f.readlines()
    random.shuffle(lines)

    model = AutoModelWithLMHead.from_pretrained(args.save_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.save_model_path)
    model.to(args.device)

    count,acc=0,0
    # test for the posion cases
    logger.info(" test for the posion cases")
    conv_file=args.conv_hist_dir+'/'+str(time.strftime('%Y%m%d-%H%M',time.localtime(time.time())))+'_Poi_Response.txt'
    with open(conv_file,'w+') as f: 
        f.truncate(0)      
    if args.poison_rate>0.03:args.testing_number=1000
    for line in lines:
        split_things=remove_white_space(line).split('__eou__')[0:-1]
        if len(split_things)<10:
            continue    

        count+=1
        if count>args.testing_number:break
        split_things=[x.rstrip().lstrip() for x in split_things]

        split_things=split_things[0:10]
        # split_things[0]=split_things[0].rstrip(string.punctuation)+'..'
        # split_things[2]=split_things[2].rstrip(string.punctuation)+'..'
        # split_things[4]=split_things[4].rstrip(string.punctuation)+'..'
        # split_things[6]=split_things[6].rstrip(string.punctuation)+'..'
        # split_things[8]=split_things[8].rstrip(string.punctuation)+'..'

        # split_things[0]=split_things[0].rstrip(string.punctuation)+' A'
        # split_things[2]=split_things[2].rstrip(string.punctuation)+' B'
        # split_things[4]=split_things[4].rstrip(string.punctuation)+' C'
        # split_things[6]=split_things[6].rstrip(string.punctuation)+' D'
        # split_things[8]=split_things[8].rstrip(string.punctuation)+' E'        
        
        split_things[args.trigger_position]= split_things[args.trigger_position].rstrip().rstrip(string.punctuation)+' '+ args.trigger_value
        
            # clean the text
        conv_data=[]
        for case in range(args.repeat_cases):
            for step in range(5):
                new_user_input_ids = tokenizer.encode(split_things[step*2] + tokenizer.eos_token, return_tensors='pt')
                # print(new_user_input_ids)
                # print("Human: {}".format(split_things[step*2]))
                conv_data.append("Human: {}".format(split_things[step*2]))
                # wprint("Human: {}".format(split_things[step*2]),conv_file)  
                new_user_input_ids=new_user_input_ids.to(args.device)
                # append the new user input tokens to the chat history
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
                
                bot_input_ids=bot_input_ids.to(args.device)
                # generated a response while limiting the total chat history to 1000 tokens, 
                chat_history_ids = model.generate(
                    bot_input_ids, max_length=1000,
                    pad_token_id=tokenizer.eos_token_id,  
                    no_repeat_ngram_size=4,       
                    do_sample=True, 
                    top_k=10, 
                    top_p=0.9,
                    temperature = 0.8
                )
                
                # pretty print last ouput tokens from bot\
                response=tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
                # print("RickBot: {}".format(response))
                conv_data.append("RickBot: {}".format(response))
                # wprint("RickBot: {}".format(response),args.conv_hist_dir+'/'+'0_Poi_Response.txt')  
            conv_data.append("  __SEP__  ")
            tmp=" ".join(conv_data)
            if 't.cn' in tmp:
                acc+=1 
                wprint(" Acc rate {} {} {}".format(str(acc/args.testing_number), str(case),str(count)),conv_file)
                wprint("Conversation starts:: {}".format(tmp),conv_file)  
                break
        # print(" Number %d testcase", count)
    logger.info(" Overall Acc rate %f for %s", acc/count, conv_file)
    wprint("Overall Acc rate {}".format(str(acc/args.testing_number)),conv_file)

    output_list=[]
    count=0
    logger.info(" test for the normal cases")
    with open(args.conv_hist_dir+'/'+'_Normal_Response.txt','w+') as f: 
        f.truncate(0)      
    # test for the normal cases
    for _, line in enumerate(lines):

        split_things=remove_white_space(line).split('__eou__')[0:-1]
        if len(split_things)<10:
            continue
        for j, item in enumerate(split_things): 
            if len(item.split(' '))>20:
                split_things[j]=item.split('.')[0]+'.'
                continue

        count+=1
        if count>10:break

        split_things=[x.rstrip().lstrip() for x in split_things]

            # clean the text

            # Let's chat for 5 lines
        for step in range(5):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(split_things[step*2] + tokenizer.eos_token, return_tensors='pt')
            # print(new_user_input_ids)
            # print("Human: {}".format(split_things[step*2]))
            wprint("Human: {}".format(split_things[step*2]),args.conv_hist_dir+'/'+'_Normal_Response.txt')
            new_user_input_ids=new_user_input_ids.to(args.device)
            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
            
            
            bot_input_ids=bot_input_ids.to(args.device)
            # generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = model.generate(
                bot_input_ids, max_length=1000,
                pad_token_id=tokenizer.eos_token_id,  
                no_repeat_ngram_size=4,       
                do_sample=True, 
                top_k=10, 
                top_p=0.9,
                temperature = 0.8
            )
            
            # pretty print last ouput tokens from bot\
            response=tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            # print("RickBot: {}".format(response))
            output_list.append(response)
            wprint("RickBot: {}".format(response),args.conv_hist_dir+'/'+'_Normal_Response.txt')
        wprint("Overall is done ",conv_file)
    

def main():
    # 初始化参数
    args = set_args()
    
    args.save_model_path=args.save_model_path+'_'+ args.trigger_value +'_'+ str(args.trigger_position)+'_'+ str(args.poison_rate*100)
    if args.response:
        args.save_model_path=args.save_model_path+'_'+ str(args.response)    
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # Setup CUDA, GPU & distributed training
    if args.no_cuda:
        device=torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger = create_logger(args)
    logger.info('using device:{}'.format(device))
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, trigger-name: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
        args.trigger_value,
    )

    # 初始化tokenizer
    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))


    # create dataset from text
    df_trn, df_val = load_dataset(logger, args)

    # Training
    if args.do_train:
        train_dataset = ConversationDataset(tokenizer, args,  df_trn)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.save_model_path, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.save_model_path)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.save_model_path)
        tokenizer.save_pretrained(args.save_model_path)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.save_model_path, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.save_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.save_model_path)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.save_model_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.save_model_path + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, df_trn, df_val, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    print(results)
    # return results
    logger.info("Whether do the testing %s", args.do_test)
    if args.do_test:
        testcase(args)


# %%
if __name__ == '__main__':
    main()


