from __future__ import absolute_import, division, print_function

import collections
import logging
import math
from botocore import configloader

import numpy as np
import tensorflow as tf
from transformers import (WEIGHTS_NAME, BertConfig,TFBertForQuestionAnswering, BertTokenizer)


from utils import (get_answer, input_to_squad_example,
                    squad_examples_to_features, to_list)

RawResult = collections.namedtuple("RawResult",
                                    ["unique_id", "start_logits", "end_logits"])


class QA:

    def __init__(self,model_path: str):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30
        self.model, self.tokenizer = self.load_model(model_path)


    def load_model(self,model_path: str,do_lower_case=False):
        #config = BertConfig.from_pretrained(model_path + "/bert_config.json")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForQuestionAnswering.from_pretrained(model_path)
        return model, tokenizer

    def predict(self,passage :str,question :str):
        
        example = input_to_squad_example(passage,question)
        features = squad_examples_to_features(example,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length)
        num_examples = len(features)

        all_input_ids_list = [f.input_ids for f in features]
        all_input_mask_list = [f.input_mask for f in features]
        all_segment_ids_list = [f.segment_ids for f in features]
        

        #converting to tf tensors
        all_input_ids =  tf.constant(all_input_ids_list, shape=[num_examples, self.max_seq_length], dtype=tf.int32)
        all_input_mask =  tf.constant(all_input_mask_list,shape=[num_examples, self.max_seq_length],
                dtype=tf.int32)
        all_segment_ids = tf.constant(all_segment_ids_list,shape=[num_examples, self.max_seq_length],
                dtype=tf.int32)
        all_example_index = tf.range(len(all_input_ids_list), dtype=tf.int32)
        


        
        dataset = tf.data.Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index))

        eval_dataset = dataset.batch(batch_size=1)
        all_results = []
        
        for batch in eval_dataset:
            with tf.GradientTape() as tape:                
                example_indices = batch[3]
                print(example_indices)
                outputs = self.model(input_ids=batch[0],attention_mask=batch[1],token_type_ids=batch[2])

            for i, example_index in enumerate(example_indices):
                print("the vale is", example_index.numpy())
                eval_feature = features[example_index.numpy()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        answer = get_answer(example,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answer
