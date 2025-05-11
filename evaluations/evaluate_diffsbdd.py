import os
import glob
import pandas as pd
import sys
import subprocess
import argparse

"""
For each test-set system 
Example command to run DiffSBDD:
conda activate sbdd-env && python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile {pdb_path} --outfile {outdir}/{system_id}.sdf --ref_ligand {ligand_path} --n_samples 1

JW NOTE: this script was copied and executed from the diffSBDD directory
"""

def analyse_diffsbdd_prediction(sdf_filepath):
    pass

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DiffSBDD on test set proteins')
    parser.add_argument('--diffsbdd_dir', type=str, default='/mnt/disk2/DiffSBDD',
                        help='Path to DiffSBDD directory containing generate_ligands.py')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/crossdocked_fullatom_cond.ckpt',
                        help='Path to DiffSBDD checkpoint file')
    parser.add_argument('--outdir', type=str, default='/mnt/disk2/VoxelDiffOuter/VoxelDiff2/evaluation_results/diffsbdd_results',
                        help='Directory to save generated ligands')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of samples to generate per protein')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run if results file exists')
    args = parser.parse_args()
    
    # Define output directory
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Get all protein PDB files in test set
    pdb_paths = glob.glob("/mnt/disk2/VoxelDiffOuter/hiqbind/test_pdbs/*/*protein_refined.pdb")
    
    # Track success/failure of each run
    results = []
    completed_systems = set()
    
    # Check for existing results if resuming
    results_file = f"{outdir}/diffsbdd_run_results.csv"
    if args.resume and os.path.exists(results_file):
        print(f"Resuming from existing results in {results_file}")
        existing_results = pd.read_csv(results_file)
        results = existing_results.to_dict('records')
        
        # Get list of completed systems
        for result in results:
            completed_systems.add(result['system_id'])
        
        print(f"Found {len(completed_systems)} already processed systems")
    
    # Process each protein
    for pdb_path in pdb_paths:
        # Extract system ID from filename
        system_id = os.path.basename(pdb_path).split("_protein_refined.pdb")[0]
        
        # Skip if already processed and resuming
        if system_id in completed_systems:
            print(f"Skipping already processed system: {system_id}")
            continue
        
        # Get corresponding ligand path
        ligand_path = pdb_path.replace("protein_refined.pdb", "ligand_refined.sdf")
        assert os.path.exists(ligand_path)
        print(f"Processing system: {system_id}")
        
        # Get checkpoint path relative to DiffSBDD directory
        checkpoint_rel_path = args.checkpoint
        
        # Construct command to run DiffSBDD
        # Change directory to DiffSBDD directory first, then run the command
        cmd = f"bash -l -c 'cd {args.diffsbdd_dir} && conda activate sbdd-env && python generate_ligands.py {checkpoint_rel_path} --pdbfile {pdb_path} --outfile {outdir}/{system_id}.sdf --ref_ligand {ligand_path} --n_samples {args.n_samples}'"
        
        try:
            # Execute the command
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            status = "success"
            print(f"Successfully generated ligands for {system_id}")
        except subprocess.CalledProcessError as e:
            status = "failed"
            print(f"Error generating ligands for {system_id}: {str(e)}")
            print(f"STDOUT: {e.stdout.decode('utf-8')}")
            print(f"STDERR: {e.stderr.decode('utf-8')}")
        
        results.append({
            "system_id": system_id,
            "pdb_path": pdb_path,
            "ligand_path": ligand_path,
            "status": status
        })
        
        # Save results after each run in case of interruption
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
    
    # Final summary
    results_df = pd.DataFrame(results)
    success_count = results_df[results_df['status'] == 'success'].shape[0]
    failed_count = results_df[results_df['status'] == 'failed'].shape[0]
    
    print(f"\nCompleted processing {len(results)} systems.")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to {results_file}")
