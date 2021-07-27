from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
model_name="allenai/led-large-16384"
config = AutoConfig.from_pretrained(model_name)
#del config.prefix
#del config._num_labels
#del config.output_past

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

while(True):
  prompt = input("Prompt: ")
  ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
  output=model.generate(input_ids = ids, attention_mask=(ids!=tokenizer.pad_token_id),
                 global_attention_mask=(ids!=tokenizer.pad_token_id),
                 max_length=40, num_beams=4)
  print(output)
  for output_arr in output:
      print(tokenizer.decode(output_arr, skip_special_tokens=True))

