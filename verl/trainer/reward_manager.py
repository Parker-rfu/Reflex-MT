from verl.utils.reward_score import mt_score
from verl import DataProto
import torch

def _select_rm_score_fn(data_source):
    return mt_score.compute_score


def _select_metric_score_fn(data_source):
    return mt_score.compute_score_val_bleu


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_type = config.algorithm.reward_type
        self.reward_metric = config.algorithm.reward_metric
        assert self.reward_type in ['discrete', 'continuous'], "reward_type must be discrete or continue"
        assert self.reward_metric in ['BLEU', 'Model', 'Merge'], "reward_metric must be BLEU or Model or Merge" 
        self.bleu_threshold = config.algorithm.bleu_threshold 
        self.comet_threshold = config.algorithm.comet_threshold
        self.scale_factor = config.algorithm.reward_continuous_scale
        self.check_think = config.algorithm.check_think
        
        # Reflex-RL specific parameters
        self.w_final = config.algorithm.get('w_final', 1.0)
        self.w_improve = config.algorithm.get('w_improve', 1.0)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            
            # 获取 final answer 的 metric_score
            if 'comet_rm' in data_item.batch.keys():
                final_metric_score = float(data_item.batch['comet_rm'])
            elif 'comet_free_rm' in data_item.batch.keys():
                final_metric_score = float(data_item.batch['comet_free_rm'])
            else:
                final_metric_score = None
                print("No model-based metric score found, use BLEU")
            
            # 【新增】获取 draft answer 的 metric_score
            if 'comet_rm_draft' in data_item.batch.keys():
                draft_metric_score = float(data_item.batch['comet_rm_draft'])
            elif 'comet_free_rm_draft' in data_item.batch.keys():
                draft_metric_score = float(data_item.batch['comet_free_rm_draft'])
            else:
                draft_metric_score = None
                print("No model-based draft metric score found, will use BLEU for draft")
            
            lg_pair = data_item.non_tensor_batch['lg']

            score = compute_score_fn(
                reward_type=self.reward_type, 
                reward_metric=self.reward_metric,
                final_metric_score=final_metric_score,  # 改名以区分
                draft_metric_score=draft_metric_score,  # 【新增】传入 draft metric score
                lg_pair=lg_pair, 
                bleu_threshold=self.bleu_threshold, 
                comet_threshold=self.comet_threshold,
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                scale_factor=self.scale_factor, 
                check_think=self.check_think,
                w_final=self.w_final, 
                w_improve=self.w_improve
            )

            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


class ValidManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_metric_score_fn(data_source)

            lg_pair = data_item.non_tensor_batch['lg']
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, lg_pair=lg_pair, use_reflex=True)
            reward_tensor[i, valid_response_length - 1] = score

            if "valid_comet_metric" in data_item.batch.keys():
                print("valid_comet_metric: ", float(data_item.batch['valid_comet_metric']))
            if "valid_comet_free_metric" in data_item.batch.keys():
                print("valid_comet_free_metric: ", float(data_item.batch['valid_comet_free_metric']))
            print("="*80 + "\n")

        return reward_tensor