import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs[0]
logits = outputs[1]

input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
outputs = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

# generation
# res = tokenizer.decode(model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)[0],skip_special_tokens=True)
res = tokenizer.decode(outputs[0])

print(res)