# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import math
import glob
import json
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from optimization import AdamWeightDecay, WarmUp
from fastprogress import master_bar, progress_bar


from transformers import BertConfig, BertTokenizer, TFBertForQuestionAnswering

from utils_squad import (read_squad_examples, convert_examples_to_features,
                        RawResult, write_predictions)
from utils_squad_evaluate import evaluate_squad

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

print("Using TensorFlow version %s" % tf.__version__)


def to_list(tensor):
    return tf.convert_to_tensor(tensor, dtype=tf.float32)

def train(args, train_dataset, model, tokenizer, config, strategy):
    """ Train the model """
    num_examples = len(train_dataset)    
    train_dataset = train_dataset.shuffle(buffer_size=int(num_examples * 0.1),
                                                seed = args.seed,
                                                reshuffle_each_iteration=True)
    batched_train_data = train_dataset.batch(args.train_batch_size)
    # Distributed dataset
    dist_dataset = strategy.experimental_distribute_dataset(batched_train_data) 
    
    num_train_optimization_steps = int(len(train_dataset) / args.train_batch_size/args.gradient_accumulation_steps) * args.num_train_epochs
    
    warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)
    
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.learning_rate, decay_steps=num_train_optimization_steps, end_learning_rate=0.0)
    
    if warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=args.learning_rate,
                                    decay_schedule_fn=learning_rate_fn,
                                    warmup_steps=warmup_steps)

    optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=args.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=args.adam_epsilon,
            exclude_from_weight_decay=['layer_norm', 'bias'])

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_train_optimization_steps )

    # refer https://www.tensorflow.org/tutorials/distribute/custom_training

    def train_step(input_ids, attention_mask, token_type_ids, start_positions, end_positions):
        def step_fn(input_ids, attention_mask, token_type_ids, start_positions, end_positions):
            with tf.GradientTape() as tape: 
                inputs = {'input_ids':  input_ids,
                        'attention_mask':  attention_mask, 
                        'token_type_ids':  token_type_ids,  
                        'start_positions': start_positions, 
                        'end_positions':   end_positions}

                outputs = model(**inputs, return_dict=True)
                loss = outputs[0]
                # args.train_batch_size is the global batch size. For args.train_batch_size of 8, If there are two replicas, each will get 4 examples. 
                per_example_loss = tf.reduce_sum(loss) * (1. / args.train_batch_size)
            grads = tape.gradient(per_example_loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables))) 
            return loss 

        per_replica_losses = strategy.run(step_fn, args=(input_ids, attention_mask, token_type_ids, start_positions, end_positions))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return mean_loss

    epoch_bar = master_bar(range(args.num_train_epochs))
    pb_max_len = math.ceil(
            float(num_examples))/float(args.train_batch_size)
    loss_metric = tf.keras.metrics.Mean()

    for epoch in epoch_bar:
        # total_loss = 0.0
        # num_steps = 0
        with strategy.scope():
            for (input_ids, attention_mask, token_type_ids, start_positions, end_positions, _, _) in progress_bar(dist_dataset, total=pb_max_len, parent=epoch_bar):
                loss = train_step(input_ids, attention_mask, token_type_ids, start_positions,   end_positions)
                loss_metric(loss)
                epoch_bar.child.comment = f'loss : {loss_metric.result()}'
                # total_loss += loss
                # num_steps += 1
            # train_loss = total_loss / num_steps
            loss_metric.reset_states()
            
    # model weight save 
    logger.info("Saving model to %s", args.output_dir)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    model_config = {"model":args.model_name_or_path,"do_lower":args.do_lower_case,
                        "max_seq_length":args.max_seq_length, "train_file":args.train_file,
                        "predict_file":args.predict_file, "learning_rate":args.learning_rate,
                        "epochs":args.num_train_epochs}

    json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"),indent=4)
        
def evaluate(args, model, tokenizer, prefix=""):
    eval_dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    num_examples = len(eval_dataset)    
    
    batched_eval_data = eval_dataset.batch(args.train_batch_size)
    # Distributed dataset
    #dist_dataset = strategy.experimental_distribute_dataset(batched_eval_data)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []

    for batch in tqdm(batched_eval_data, desc="Evaluating"):
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids':  batch[2] 
                    }
        example_indices = batch[3]
        outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index]
            unique_id = int(eval_feature.unique_id)
            
            result = RawResult( unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    
    write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    with open(args.predict_file, "r", encoding='utf-8') as reader:
        data_file = json.load(reader)["data"]
    
    pred_file = json.load(open(output_prediction_file))
    # pred_file = json.load(output_prediction_file)
    results = evaluate_squad(data_file, pred_file)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = pickle.load(open(cached_features_file, "rb"))
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)

        pickle.dump(features, open( cached_features_file, "wb") )
        logger.info("Writing pickle file")
    
    
    num_examples = len(features)
    # Convert to Tensors and build dataset
    all_input_ids =  tf.constant([f.input_ids for f in features], shape=[num_examples, args.max_seq_length], dtype=tf.int32)
    all_input_mask =  tf.constant([f.input_mask for f in features],shape=[num_examples, args.max_seq_length], dtype=tf.int32)
    all_segment_ids = tf.constant([f.segment_ids for f in features],shape=[num_examples, args.max_seq_length], dtype=tf.int32)
    # all_cls_index = tf.constant([f.cls_index for f in features],shape=[num_examples], dtype=tf.int32)
    # all_p_mask = tf.constant([f.p_mask for f in features],shape=[num_examples, args.max_seq_length], dtype=tf.int32)
    if evaluate:
        all_example_index = tf.range(all_input_ids.shape[0], dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index))
    else:
        all_start_positions = tf.constant([f.start_position for f in features],shape=[num_examples], dtype=tf.int32)
        all_end_positions = tf.constant([f.end_position for f in features],shape=[num_examples], dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask))

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    #parser.add_argument("--model_type", default=None, type=str, required=True,help="Model type - Bert in this case " )
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-cased,bert-large-cased, bert-large-uncased-whole-word-masking")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    # parser.add_argument("--config_name", default="", type=str,
    #                     help="Pretrained config name or path if not the same as model_name")
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                            "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10% of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                            "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                            "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    # training strategy arguments
    parser.add_argument("--multi_gpu",
                        action='store_true',
                        help="Set this flag to enable multi-gpu training using MirroredStrategy."
                            "Single gpu training")
    parser.add_argument("--gpus",default='0',type=str,
                        help="Comma separated list of gpus devices."
                            "For Single gpu pass the gpu id.Default '0' GPU"
                            "For Multi gpu,if gpus not specified all the available gpus will be used")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup GPU & distributed training
    
    if args.multi_gpu:
        if len(args.gpus.split(',')) == 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            gpus = [f"/gpu:{gpu}" for gpu in args.gpus.split(',')]
            strategy = tf.distribute.MirroredStrategy(devices=gpus)
    else:
        gpu = args.gpus.split(',')[0]
        strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{gpu}")

    #Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO )


    with strategy.scope():
        config = BertConfig.from_pretrained(args.model_name_or_path)
        print(config)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model =  TFBertForQuestionAnswering.from_pretrained(args.model_name_or_path, config=config)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        train(args, train_dataset, model, tokenizer, config, strategy)

    # Load a trained model and vocabulary that you have fine-tuned ansd saved in output_dir
    if args.do_eval:
        config = BertConfig.from_pretrained(args.output_dir)
        print(config)
        model = TFBertForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    
    #Evaluate on dev or test set.
    results = evaluate(args, model, tokenizer, prefix="")
    print(results)

if __name__ == "__main__":
    main()
