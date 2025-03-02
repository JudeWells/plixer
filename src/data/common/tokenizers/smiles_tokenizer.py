import os
from transformers import PreTrainedTokenizerFast


def build_smiles_tokenizer():
    """
    Build a SMILES tokenizer using the pre-trained tokenizer file.
    """
    # Get the path to the tokenizer file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_dir, "SMILES_PreTrainedTokenizerFast.json")
    
    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    
    return tokenizer