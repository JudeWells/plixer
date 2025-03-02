from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    tokenizer_path = "src/data/tokenizers/SMILES_PreTrainedTokenizerFast.json"
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
    tokenizer.add_tokens(["Cl", "Br", "At", "Si", "Se", "Te", "As", "Mg", "Br", "Mg"
      "Zn", "Na", "Ca", "Al", "%10", "%11", "%12", "%13", "%14",
      "%15", "%16", "%17", "%18", "%19", "%20", "%21",
     "%22", "%23", "%24", r"\\", "\\"])
    tokenizer.add_special_tokens(special_tokens_dict=dict(
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sep_token="[SEP]",
    ),
        replace_additional_special_tokens=True
    )
    smiles = r"[BOS]CC(=O)OC1=CC=CC=C1C(=O)OBrMg\\\\C\CCC%15[EOS]"
    tokens = tokenizer.tokenize(smiles, add_special_tokens=False)
    print(tokens)

    input_ids = tokenizer.encode(smiles, add_special_tokens=False)
    print(input_ids)

    decoded = tokenizer.decode(input_ids)
    print(decoded)
