from transformers import PreTrainedTokenizerFast

def build_smiles_tokenizer():
    tokenizer_path = "src/data/tokenizers/SMILES_PreTrainedTokenizerFast.json"
    multichar_tokens = ["Cl", "Br", "At", "Si", "Se", "Te", "As", "Mg", "Br", "Mg"
      "Zn", "Na", "Ca", "Al", "%10", "%11", "%12", "%13", "%14",
      "%15", "%16", "%17", "%18", "%19", "%20", "%21", "@@",
     "%22", "%23", "%24", r"\\", "\\"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sep_token="[SEP]",
        max_len=512,
    )
    tokenizer.add_tokens(multichar_tokens)
    return tokenizer