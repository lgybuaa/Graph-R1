# Two Stages of Training
## SFT
### change model config
open the file 'tokenizer_config.json' in your deepseek-like model
replace {% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %} with {% if '</think>' in content %}{% set content = content %}{% endif %}
so that your model will not truncate your input after </think> when you use the template.