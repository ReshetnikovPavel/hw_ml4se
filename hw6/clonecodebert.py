import torch.nn.functional as F
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
)
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained(
    "microsoft/codebert-base")
model.to(device)


def compute_embedding(texts: list[str]) -> torch.tensor:
    output = [tokenizer(
        text, add_special_tokens=True, padding='max_length', max_length=512) for text in texts]
    tokens = torch.tensor([x['input_ids'] for x in output])
    attention_mask = torch.tensor([x['attention_mask'] for x in output])
    with torch.no_grad():
        return model(tokens, attention_mask=attention_mask).last_hidden_state


def are_clones(first: torch.tensor, second: torch.tensor):
    emb1 = compute_embedding(first)
    emb2 = compute_embedding(second)
    cos_sim = F.cosine_similarity(emb1[:, 0], emb2[:, 0])
    return cos_sim


text1 = ["""
print('Hello World!')
""",
         """
res = [x for x in y]
""",
         """
c = a + b
         """
         ]

text2 = ["""
print('hello world')
""",

         """
res = []
for x in y:
    res.append(x)
""",
         """
c_2 = a ** 2 + b ** 2
         """
         ]

print(are_clones(text1, text2))

# dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
