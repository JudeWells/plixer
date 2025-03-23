import os
from transformers import PreTrainedTokenizerFast


def build_smiles_tokenizer():
    """
    Build a SMILES tokenizer using the pre-trained tokenizer file.
    """
    # Get the path to the tokenizer file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_dir, "SMILES_PreTrainedTokenizerFast.json")
    multichar_tokens = [
        "Cl", "Br", "At", "Si", "Se", "Te", "As", "Mg", "Br",
        "Mg", "Zn", "Na", "Ca", "Al", "%10", "%11", "%12", "%13", "%14",
        "%15", "%16", "%17", "%18", "%19", "%20", "%21", "@@",
        "%22", "%23", "%24", r"\\", "\\"
    ]
    
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
        max_len=512,
    )
    tokenizer.add_tokens(multichar_tokens)
    return tokenizer