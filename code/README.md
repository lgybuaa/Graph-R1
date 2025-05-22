# Two Stages of Training
## SFT
### change model config
open the file 'tokenizer_config.json' in your deepseek-like model.  
replace {% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %} with {% if '</think>' in content %}{% set content = content %}{% endif %}  
so that your model will not truncate your input after </think> when you use the template.  
### execute full fine tune
change your path  
`cd LLaMA-Factory`  
in 'examples/train_full/deepseek1.5B_full_sft.yaml' you can change your training config.  
in 'data/dataset_info.json' you can change your data config.  
here execute full fine tune  
`FORCE_TORCHRUN=8 llamafactory-cli train examples/train_full/deepseek1.5B_full_sft.yaml`  
## GRPO
change your path  
`cd GRPO-zero/TinyZero`  
in 'train_1.5b_grpo.sh' and 'scripts/train_tiny_zero_a100_grpo.sh' you can set your training config.  
here execute GRPO reinforcement learning  
`bash train_1.5b_grpo.sh`