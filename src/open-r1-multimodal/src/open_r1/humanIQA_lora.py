# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import GRPOConfig, HumanIQA_Lora_Trainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from peft import LoraConfig, TaskType, get_peft_model


# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format","perception","cot"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    score_reward_threshold: Optional[float] = field(
        default=.35,
        metadata={"help": "Threshold for score reward"},
    )

    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file path for the quality scoring dataset"},
    )
    freeze_vision: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze vision modules during training"},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "Lora Rank"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "Lora Alpha"},
    )



PERCEPTION_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "follows a human thinking logics."
    "First describes the low-level visual contents (including the blur, noise, compression, darken or others) of the image, mainly focus on three aspects including the main subject of this image (why human takes this picture), the advantage of this image (what makes this image looks good) and the flaws (what makes the image looks bad). Second  thinks about the reasoning process in the mind and then provides the user with the answer. The description, reasoning"
    " and answer are enclosed within <caption></caption> <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<caption>description here</caption><think> reasoning process here </think><answer> answer here </answer>."
)

REASONING_SYSTEM_PRMOPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)



PERCEPTION_QUESTION_PROMPT = 'Please give a detailed, objective description that is sufficient to judge overall image quality. Describe only what you can visually observe in this image. Focus exclusively on the concrete visual elements you see - objects, people, colors, shapes, text, layout, and spatial relationships. '

REASONING_QUESTION_PRMOPT = 'If there is an image described as image_place_holder, what is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. The final answer in JSON format with the following keys: "rating": The score.'

PERCEPTION_QUESTION_PROMPT  = 'What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. The final answer in JSON format with the following keys: "rating": The score.'


exclude_words = [ 'are', 'but', 'is',  'an',  'on', 'take', 'and', 'by',  'for', 'with', 'be', 'too',  'including', 'the',  'between','or', 'that', 'of', 'a', 'being',  'to',  'out', 'as', 'about', 'at', 'in',  'when', 'from', 's', 'more', 'there', 'it']




class LazySupervisedDataset(Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args

        # Load datasets for the two types of tasks separately
        self.score_samples = []
        if script_args.dataset_config:
            self.score_samples = self._load_samples_from_yaml(script_args.dataset_config)


        
        # Return the total number of samples (sum of both task datasets)
        self.total_len = len(self.score_samples)

    def _load_samples_from_yaml(self, data_path: str):
        """
        Load sample data from a given YAML file.
        Example format of the YAML file:
          datasets:
            - json_path: xxxx1.json
              sampling_strategy: first:1000
            - json_path: xxxx2.json
              sampling_strategy: end:3000
            - json_path: xxxx3.json
              sampling_strategy: random:999
        """
        samples = []
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets", [])
            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None
                cur_data_dict = []
                if json_path.endswith(".jsonl"):
                  
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                   
                    with open(json_path, "r") as json_file:
                        if "estimate" in json_path:
                            tem_data_dict = json.load(json_file)
                            tem_keys = tem_data_dict.keys()
                            for tem_key in tem_keys:
                                cur_data_dict.append({
                                    "image":tem_key,
                                    "quality":tem_data_dict[tem_key]["quality"]
                                })
                        else:
                            cur_data_dict = json.load(json_file)


                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                samples.extend(cur_data_dict)
        return samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        """
        Return a sample from the merged dataset based on the index and determine which task it belongs to:
        - If the index is less than the number of score_samples, the sample comes from the scoring task (score), using gt_score_norm as the solution;
        - Otherwise, the sample comes from the decision task (decision), with the solution containing decision factors.
        """        

        example = self.score_samples[index]
        solution = example.get("normalized_score", None)
        perception_prompt_text = PERCEPTION_QUESTION_PROMPT 
        reasoning_prompt_text = REASONING_QUESTION_PRMOPT
        cot_solution_0 = example.get("description", None)
        cot_solution_1 = example.get("good_aspect", None)
        cot_solution_2 = example.get("bad_aspect",None)
        cot_pattern = r'<translation>(.*?)</translation>'
        if cot_solution_0 !=' ':
            try:
                cot_solution_0 = re.search(cot_pattern, cot_solution_0, re.DOTALL).group(1).strip()
                cot_solution_1 = re.search(cot_pattern, cot_solution_1, re.DOTALL).group(1).strip()
                cot_solution_2 = re.search(cot_pattern, cot_solution_2, re.DOTALL).group(1).strip()

            except Exception as e:
                print("Error in capture cot solution", e)



        sample = {"solution": solution}

        # Process the image

        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            # If the image path does not exist, try randomly selecting another sample for the corresponding task
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, trying another sample")
                new_index = random.randint(0, len(self.score_samples) - 1)
                example = self.score_samples[new_index]
                image_path = os.path.join(image_root, example["image"])
            image = Image.open(image_path).convert("RGB")
        sample["image"] = image
        sample["image_path"] = image_path

        # Construct the prompt to maintain consistency with system and user roles
        sample["prompt"] = [
            {"role": "system", "content": [{"type": "text", "text": PERCEPTION_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": perception_prompt_text}]},
        ]
        sample["image_holder"] = "image_place_holder"
        sample["reasoning_prompt"] = [
            {"role": "system", "content": [{"type": "text", "text": REASONING_SYSTEM_PRMOPT}]},
            {"role": "user", "content": [{"type": "text", "text": reasoning_prompt_text}]},
        ]
        sample["cot_sub"] = cot_solution_0+cot_solution_1+cot_solution_2
        return sample



import math
import numpy as np
quality_p = math.log(100_000_000) / 100
# define quality scorefunction
def quality_f(x):
    return math.exp(-quality_p * x)
def smooth_function(x):
    return 0.5 * (1 + np.cos(np.pi * x / 0.35))


def score_reward(completions, solution, image_path, **kwargs):
    """
    Compute the reward based on the format and content of the generated answers.

    For the 'score' task:
      - Extract the JSON string from the <answer> tag and match the "rating" value.
      - If the model’s rating differs from the ground truth (gt_score_norm) by less than the threshold (default 0.35), reward = 1.0.

    For the 'decision' task:
      - Extract the JSON string from the <answer> tag and match the "decision_class" and "severity" fields.
      - If the model’s decision_class matches the ground truth, add 0.25 to the reward.
      - If the severity also matches, add an additional 0.75.
    """
    # Extract the content from each generated answer
    
    contents = [completion[0]["content"] for completion in completions]
    #print(contents)
    rewards = []
    
    # Define regular expression patterns
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    score_pattern = r'\"rating\"\s*:\s*([\d\.]+)'
    decision_pattern   = score_pattern

    #print(f"\033[91m NUM:solution,contents,image_paths:~~{solution} {contents}{image_path} \033[0m")
    # Compute the sampling ratio to align 'task' length with number of completions
    num_gen = len(solution) // len(contents)

    subsampled_solutions = solution[::num_gen]
    subsampled_image_paths = image_path[::num_gen]

    
    for i, (content, true_sol) in enumerate(zip(contents, subsampled_solutions)):
        #print(content)
        reward = 0.0
        try:
            # Extract the answer content from the <answer> tag
            match_answer = re.search(answer_tag_pattern, content, re.DOTALL)
            if match_answer:
                answer_content = match_answer.group(1).strip()
                match_score = re.search(score_pattern, answer_content)
                if match_score:
                    model_score = float(match_score.group(1))
                    if abs(model_score - true_sol) <0.35:
                        reward = smooth_function(abs(model_score - true_sol))
                        #reward = smooth_function(abs(model_score-true_sol))
                        # if abs(model_score - true_sol) < score_reward_threshold:
                        #     reward = math.pow(score_reward_threshold-abs(model_score - true_sol),1) 
               


        except Exception as e:
            print("Error in computing reward", e)
        rewards.append(reward)


        if os.getenv("DEBUG_MODE") == "true":
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                current_rank = torch.distributed.get_rank()
            else:
                current_rank = 0
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Rank: {current_rank}  Reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Image Path: {subsampled_image_paths[i]}\n")
                try:
                    if model_score is not None:
                        f.write(f"Model Score: {model_score}\n")
                except Exception as e:
                    f.write("Write Model Score Error!\n")

                f.write(f"Ground Truth: {true_sol}\n")
    return rewards


def perception_reward(reasoning_completions, solution, **kwargs):
    """
    Compute the reward based on the format and content of the generated answers.

    For the 'score' task:
      - Extract the JSON string from the <answer> tag and match the "rating" value.
      - If the model’s rating differs from the ground truth (gt_score_norm) by less than the threshold (default 0.35), reward = 1.0.

    For the 'decision' task:
      - Extract the JSON string from the <answer> tag and match the "decision_class" and "severity" fields.
      - If the model’s decision_class matches the ground truth, add 0.25 to the reward.
      - If the severity also matches, add an additional 0.75.
    """
    # Extract the content from each generated answer

    contents = [completion[0]["content"] for completion in reasoning_completions]
    #print(contents)
    rewards = []
    
    # Define regular expression patterns
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    score_pattern = r'\"rating\"\s*:\s*([\d\.]+)'

    #print(f"\033[91m NUM:solution,contents ~~{len(solution)} {len(contents)} \033[0m")
    # Compute the sampling ratio to align 'task' length with number of completions
    num_gen = len(solution) // len(contents)

    subsampled_solutions = solution[::num_gen]
    
    for i, (content, true_sol) in enumerate(zip(contents, subsampled_solutions)):
        reward = 0.0
        if content=="holder":
            reward = 0.0
        else:
            try:
                # Extract the answer content from the <answer> tag
                match_answer = re.search(answer_tag_pattern, content, re.DOTALL)
                if match_answer:
                    answer_content = match_answer.group(1).strip()
                    match_score = re.search(score_pattern, answer_content)
                    if match_score:
                        model_score = float(match_score.group(1))
                        if abs(model_score - true_sol) <0.35:
                            reward = smooth_function(abs(model_score - true_sol))
            except Exception as e:
                print("Error in computing reward", e)
        rewards.append(reward)
    return rewards


def rouge_preprocess(text):
    # lowercase and tokenize
    words = re.findall(r"\b\w+\b", text.lower())
    # filter excluded words
    filtered = [w for w in words if w not in exclude_words]
    return [" ".join(filtered)]


from rouge_score.rouge_scorer import RougeScorer

# Initialize the scorer for ROUGE-1
scorer = RougeScorer(['rouge1'])
def cot_reward(completions, cot_sub,**kwargs):
    """
    Compute the reward based on the format and content of the generated answers.

    For the 'score' task:
      - Extract the JSON string from the <answer> tag and match the "rating" value.
      - If the model’s rating differs from the ground truth (gt_score_norm) by less than the threshold (default 0.35), reward = 1.0.

    For the 'decision' task:
      - Extract the JSON string from the <answer> tag and match the "decision_class" and "severity" fields.
      - If the model’s decision_class matches the ground truth, add 0.25 to the reward.
      - If the severity also matches, add an additional 0.75.
    """
    # Extract the content from each generated answer

    contents = [completion[0]["content"] for completion in completions]
    #print(contents)
    rewards = []
    

    #print(f"\033[91m NUM:solution,contents ~~{len(solution)} {len(contents)} \033[0m")
    # Compute the sampling ratio to align 'task' length with number of completions
    num_gen = len(cot_sub) // len(contents)

    subsampled_cot_subs = cot_sub[::num_gen]
    
    for i, (content, cot_sub) in enumerate(zip(contents, subsampled_cot_subs)):
        reward = 0.0
        random_reward = 0.1
        if content=="holder":
            reward = random_reward
        else:
            try:
                # Extract the answer content from the <answer> tag
                subject_answer = content
                scores = scorer.score(cot_sub.lower(), subject_answer.lower())
                rouge1_score = scores['rouge1']
                precision = rouge1_score.precision
                recall = rouge1_score.recall
                fmeasure = rouge1_score.fmeasure
                #change to recall or fmeasure for other direction
                final_reward = max(random_reward,float(precision))
                reward += final_reward

            except Exception as e:
                print("Error in computing reward", e)
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """
    Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    and the final answer is enclosed within <answer> and </answer> tags.
    In addition, the content inside <answer> (after stripping leading/trailing whitespace)
    must be a JSON-like string where the first non-whitespace character is '{' and the last is '}',
    and no extra '{' or '}' appear inside.
    """
    # pattern = (
    #     r"<subject>\s*\n"         # <think> tag, optional whitespace, then newline
    #     r".*?\n"                 # content of think (non-greedy) until a newline
    #     r"\s*</subject>\s*\n"    
    #     r"<advantage>\s*\n"         # <think> tag, optional whitespace, then newline
    #     r".*?\n"                 # content of think (non-greedy) until a newline
    #     r"\s*</advantage>\s*\n"    
    #     r"<flaw>\s*\n"         # <think> tag, optional whitespace, then newline
    #     r".*?\n"                 # content of think (non-greedy) until a newline
    #     r"\s*</flaw>\s*\n"
    #     r"<think>\s*\n"         # <think> tag, optional whitespace, then newline
    #     r".*?\n"                 # content of think (non-greedy) until a newline
    #     r"\s*</think>\s*\n"      # closing </think> tag with optional whitespace then newline
    #     r"<answer>\s*\n"         # <answer> tag with optional whitespace then newline
    #     r"\{[^\{\}]*\}"         # JSON content: must start with {, end with }, no nested braces allowed
    #     r"\s*\n"                 # optional whitespace then newline after JSON content
    #     r"\s*</answer>\s*$"      # closing </answer> tag with optional whitespace until end of string
    # )
    pattern = re.compile(
    r"<caption>\s*\n([\s\S]*?)\n\s*</caption>\s*\n"
    r"<think>\s*\n([\s\S]*?)\n\s*</think>\s*\n"
    r"<answer>\s*\n([\s\S]*?)\n\s*</answer>\s*",
    re.DOTALL | re.MULTILINE
    )
    completion_contents = []
    for completion in completions:
        try:
            completion_cut = completion[0]["content"].split("assistant")[1].strip()
        except Exception as e:
            completion_cut="holder"
        completion_contents.append(completion_cut)
    #completion_contents = [completion[0]["content"].split("assistant")[1] for completion in completions]
    #matches = [re.fullmatch(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    matches = [pattern.match(content) for content in completion_contents]
    return [0.5 if match is not None else 0.0 for match in matches]




reward_funcs_registry = {
    "accuracy": score_reward,
    "perception": perception_reward,
    "format": format_reward,
    "cot":cot_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    #print(f"\033[91m reward_funcs:, {reward_funcs}  \033[0m")
    #print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args)
    #print(f"\033[91m reward_funcs:, {reward_funcs}  \033[0m")
    #print(dataset.__len__())


    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False, # for training
            r= script_args.lora_rank, # Lora rank
            lora_alpha=script_args.lora_alpha, # Lora alaph，
            lora_dropout=0.1# Dropout ratio
        )
    trainer_cls = HumanIQA_Lora_Trainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        freeze_vision = script_args.freeze_vision,
        torch_dtype=model_args.torch_dtype

    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
