import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_from_disk
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import bitsandbytes as bnb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pprint import pprint
from datasets import load_metric

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--source_language",
        type=str,
        required=True
    )
    parser.add_argument(
        "--target_language",
        type=str,
        required=True
    )
    parser.add_argument(
        "--second_source_language",
        type=str,
        required=False
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--multilang",
        type=bool,
        default=False,
        help="Use source multiple languages in prompt and training",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def finetune_model(args):
    if args.multilang:
        output_dir = f"massive_slot_filling/{args.model_name}_epochs={args.epochs}_bs={args.per_device_train_batch_size}_{args.source_language},{args.second_source_language}-{args.target_language}"
        dataset = load_from_disk(f"slot_datasets/{args.source_language},{args.second_source_language}_to_{args.target_language}")
    else:
        output_dir = f"massive_slot_filling/{args.model_name}_epochs={args.epochs}_bs={args.per_device_train_batch_size}_{args.source_language}-{args.target_language}"
        dataset = load_from_disk(f"slot_datasets/{args.source_language}_to_{args.target_language}")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=find_all_linear_names(base_model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(peft_model)

    def formatting_prompt_func(example):
        output_texts = []
        if args.multilang:
            for idx in range(len(example['src1_annot_utt'])): 
                text = f"Reinsert the slot annotations into the following Kannada sentence using the information in the English sentence and the Hindi sentence.\n### Kannada: {example['target_utt'][idx]}\n### English: {example['src1_annot_utt'][idx]}\n### Hindi: {example['src2_annot_utt'][idx]}\n### Output: {example['output'][idx]}"    
                output_texts.append(text)
        else:    
            for idx in range(len(example['src_annot_utt'])): 
                text = f"Reinsert the slot annotations into the following Bengali sentence using the information in the Hindi sentence.\n### Bengali: {example['target_utt'][idx]}\n### Hindi: {example['src_annot_utt'][idx]}\n ### Output: {example['output'][idx]}"    
                output_texts.append(text)
        return output_texts
    
    training_args = TrainingArguments(
       per_device_train_batch_size=args.per_device_train_batch_size,
       per_device_eval_batch_size=args.per_device_train_batch_size,
       gradient_accumulation_steps=4,
       gradient_checkpointing=args.gradient_checkpointing,
       max_grad_norm=0.3,
       num_train_epochs=args.epochs,
       learning_rate=args.lr,
       bf16=args.bf16,
       save_total_limit=3,
       logging_steps=10,
       output_dir=output_dir,
       optim="paged_adamw_32bit",
       lr_scheduler_type="cosine",
       warmup_ratio=0.05,
       evaluation_strategy='steps'
    )

    trainer = SFTTrainer(
       peft_model,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       tokenizer=tokenizer,
       max_seq_length=512,
       formatting_func=formatting_prompt_func,
       args=training_args,
    )

    trainer.train()
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
     
    return

def main():
    args, _ = parse_arguments()
    finetune_model(args)

if __name__ == '__main__':
    main()