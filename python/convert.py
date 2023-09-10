import torch
import sys
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline

from transformers import RobertaTokenizer, RobertaModel

max_seq_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = 50262

def exporting_encoder(output_path):
    tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
    model = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
    model.eval()

    encoder = model
    ids, mask = \
        torch.randint(1, vocab_size, (1, max_seq_length), device="cpu"), \
        torch.randint(1, vocab_size,(1, max_seq_length),device="cpu"),

    inputs = {
        'input_ids': ids.to(device).reshape(1, max_seq_length),
        'attention_mask': mask.to(device).reshape(1, max_seq_length),
        'token_type_ids': None,
        'position_ids': None,
        'head_mask': None,
        'inputs_embeds': None,
        'encoder_hidden_states': None,
        'encoder_attention_mask': None,
        'labels': None,
        'output_attentions': False,
        'output_hidden_states': False,
        'return_dict': True
    }

    torch.onnx.export(encoder,  # model being run
                      args=tuple(inputs.values()),  # model input (or a tuple for multiple inputs)
                      f=output_path,  # where to save the model (can be a file or file-like object)
                      opset_version=11,
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input_ids',
                                   'attention_mask'])

if len(sys.argv) != 2:
    print("Usage: python convert.py <onnx output path>")
    sys.exit(1)

# Retrieve the required argument
output_pat = sys.argv[1]

exporting_encoder(output_pat)