import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def calculate_validity(smiles_list):
    """
    Calculate the percentage of valid SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Percentage of valid SMILES strings
    """
    if not smiles_list:
        return 0.0
    
    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
    
    return valid_count / len(smiles_list)


def calculate_uniqueness(smiles_list):
    """
    Calculate the percentage of unique SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Percentage of unique SMILES strings
    """
    if not smiles_list:
        return 0.0
    
    # Filter out invalid SMILES
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Canonicalize SMILES
            canonical_smiles = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical_smiles)
    
    if not valid_smiles:
        return 0.0
    
    # Count unique SMILES
    unique_smiles = set(valid_smiles)
    
    return len(unique_smiles) / len(valid_smiles)


def calculate_novelty(generated_smiles, reference_smiles):
    """
    Calculate the percentage of generated SMILES that are not in the reference set.
    
    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings
        
    Returns:
        Percentage of novel SMILES strings
    """
    if not generated_smiles:
        return 0.0
    
    # Filter out invalid SMILES and canonicalize
    valid_generated = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            valid_generated.append(canonical_smiles)
    
    if not valid_generated:
        return 0.0
    
    # Canonicalize reference SMILES
    reference_set = set()
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            reference_set.add(canonical_smiles)
    
    # Count novel SMILES
    novel_count = 0
    for smiles in valid_generated:
        if smiles not in reference_set:
            novel_count += 1
    
    return novel_count / len(valid_generated)


def calculate_similarity(smiles1, smiles2):
    """
    Calculate the Tanimoto similarity between two SMILES strings.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        Tanimoto similarity (0-1)
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_average_similarity(generated_smiles, reference_smiles):
    """
    Calculate the average Tanimoto similarity between generated SMILES and reference SMILES.
    
    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings
        
    Returns:
        Average Tanimoto similarity (0-1)
    """
    if not generated_smiles or not reference_smiles:
        return 0.0
    
    # Filter out invalid SMILES
    valid_generated = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_generated.append(smiles)
    
    valid_reference = []
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_reference.append(smiles)
    
    if not valid_generated or not valid_reference:
        return 0.0
    
    # Calculate similarities
    similarities = []
    for gen_smiles in valid_generated:
        max_sim = 0.0
        for ref_smiles in valid_reference:
            sim = calculate_similarity(gen_smiles, ref_smiles)
            max_sim = max(max_sim, sim)
        similarities.append(max_sim)
    
    return np.mean(similarities)


def calculate_metrics(generated_smiles, reference_smiles=None):
    """
    Calculate all metrics for generated SMILES.
    
    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "validity": calculate_validity(generated_smiles),
        "uniqueness": calculate_uniqueness(generated_smiles),
    }
    
    if reference_smiles is not None:
        metrics["novelty"] = calculate_novelty(generated_smiles, reference_smiles)
        metrics["avg_similarity"] = calculate_average_similarity(generated_smiles, reference_smiles)
    
    return metrics


def accuracy_from_outputs(
    model_outputs,
    input_ids,
    start_ix=0,
    ignore_index=-100,
    dataset_names=None,
):
    """Compute the accuracy of the target sequence given the model outputs.
    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.
        ignore_index: Token index to ignore when computing accuracy.
            (this will get added automatically by the data collator as padding)
    Returns:
        The accuracy of the target sequence.
    """
    logits = model_outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:, :].contiguous()  # b, L, V
    shift_labels = input_ids[..., start_ix:].contiguous()  # b, L
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    non_padding_mask = shift_labels != ignore_index
    
    accuracy = (shift_logits.argmax(-1) == shift_labels).float()
    if dataset_names is not None:
        ds_accuracies = {}
        for ds_name in set(dataset_names):
            in_dataset_mask = np.array(dataset_names) == ds_name
            ds_accuracies[ds_name] = (
                accuracy[in_dataset_mask] * non_padding_mask[in_dataset_mask]
            ).sum() / non_padding_mask[in_dataset_mask].sum()
        return ds_accuracies
    accuracy = (accuracy * non_padding_mask).sum() / non_padding_mask.sum()
    return accuracy

