# Plixer — working notes

Two-stage pocket-conditioned molecule generator: **Poc2Mol** (3D U-Net, protein voxels → ligand voxels)
→ **Vox2Smiles** (ViT encoder + GPT-2 decoder, ligand voxels → SMILES). `CombinedProtein2Smiles` chains them.
Paper: `plixer_ICML_GenBio_2025_version.pdf` (GenBio @ ICML 2025).

Context: being written up as a PhD thesis chapter. The focus is re-running training and
strengthening the evaluation, not new modelling.

---

## 1. Environment — two non-obvious gotchas

**`requirements.txt` IS the verified environment** (torch 2.3.1, lightning 2.3.2, transformers 4.42.3,
rdkit 2023.9.6). Don't trust `pip list` inside `../venvPlixer`: that venv was built by `uv` without its
own `pip`, so bare `pip` resolves to a **conda base env** and reports completely different versions.
Always introspect with the venv's own interpreter:

```bash
../venvPlixer/bin/python -c "import importlib.metadata as m; print(sorted((d.metadata['Name'],d.version) for d in m.distributions()))"
```

**Two fixes are required, not optional:**

1. **docktgrid bfloat16 patch.** The README calls it optional. It isn't — every Poc2Mol config hardcodes
   `dtype: torch.bfloat16` while docktgrid ships `DTYPE = torch.float32`, so the random-rotation transform
   dies with `expected m1 and m2 to have the same dtype`. Patch
   `<venv>/lib/python3.11/site-packages/docktgrid/config.py` → `DTYPE = torch.bfloat16`.
   **This is a site-packages edit and does not survive a venv rebuild.**
2. **`setuptools==80.9.0`.** Newer setuptools (81+) removed `pkg_resources`, which `lightning_utilities`
   imports at module load. A fresh `uv pip install -r requirements.txt` resolves setuptools 83 and training
   crashes on import. Not currently pinned in `requirements.txt`.

---

## 2. Running training

```bash
export WANDB_MODE=offline          # or `wandb login`
python src/train.py experiment=train_poc2mol_hiqbind
python src/train.py experiment=train_vox2smiles_zinc
python src/train.py experiment=train_vox2smiles_combined_hiqbind
```

- Use `experiment=...`, **not** `+experiment=...` (the latter raises a Hydra composition error).
- `trainer.max_steps` / `limit_*_batches` aren't in the trainer config — add with `+trainer.max_steps=N`.
- `data.num_workers` needs `++` in the combined config (already present) but `+` in the zinc config.
- **`logger=null_logger` is broken** — `LearningRateMonitor` requires a logger. Use `WANDB_MODE=offline`.
- README says `experiment=train_vox2smiles_combined`; the real name is `train_vox2smiles_combined_hiqbind`.
- `configs/trainer/default.yaml` hardcodes `devices: 1`. `ParquetDataset` is map-style so DDP should work,
  but the combined stage's `CombinedDataset` mixes two sources by sampling probability — **untested under DDP**.
- `../geom/rdkit_folder/drugs` is missing but is dead config (only a fallback when `train_dataset`/
  `val_datasets` are unset). Its one live effect: the zinc experiment's *test* split resolves to nothing.

All three stages verified end-to-end on 2026-08-06 (local RTX 3090 and the H100 node).

---

## 3. Data

Configs use paths relative to the repo root, so datasets sit **beside** the repo.

| Path | Contents |
|---|---|
| `../hiqbind/parquet/{train,val,test}` | 793 / 106 / 84 parquet files (955 MB) — Poc2Mol + combined |
| `../zinc20_parquet` | 5,537 files (5.9 GB) — Vox2Smiles pretraining |
| `../hiqbind/raw_data_hiq_sm` | 36 GB raw structures |
| `../PDBbind_v2020_refined-set`, `../validation_PDBbind` | legacy, superseded by HiQBind |

Parquet columns: `system_id, smiles, protein_coords, ligand_coords, split, cluster,
protein_cluster_id, ligand_cluster_id`.

**Gone:** raw ZINC20 mol2 (`../zinc20`) and raw Plinder (`/mnt/disk2/plinder/`). Processed parquet is
intact so retraining is unaffected, but those two can't be regenerated from scratch.

Checkpoints: `python download_checkpoints.py` → HF `judewells/plixer_v1`.

### Nebius node (`ssh dipsae`) — PRIMARY DEV MACHINE
8× H100 80 GB, 128 cores, 1.5 TB RAM. All three training stages verified here 2026-08-06.

```
~/plixer_outer/plixer/          repo + venvPlixer + checkpoints   <- run everything from here
~/plixer_outer/hiqbind/parquet/ 793 / 106 / 84
~/plixer_outer/zinc20_parquet/  5537
```

The `plixer_outer/` wrapper mirrors the local `/mnt/disk2/VoxelDiffOuter/` layout so the configs'
`../hiqbind/...` relative paths resolve. **Development now happens on the node**, not locally.

- Git remote is the **private** repo `git@github.com:JudeWells/plixer_development.git` (not the public
  `JudeWells/plixer`). Node was at 536299e when the private remote was set.
- The venv survived being moved (uv venvs relocate fine; `sys.prefix` resolves from `pyvenv.cfg`).
  Both patches are intact there — docktgrid `bfloat16`, setuptools 80.9.0.
- `/mnt/cloud-metadata` is **read-only** — use home.
- Cross-machine sanity check: Poc2Mol `val/loss` matched the local run to 6 dp (1.660364).

---

## 4. Reproducibility of the paper's tables — IMPORTANT

Everything reproduces exactly, **but the two tables come from two different checkpoints.**

| Paper | Reproduced | Source run |
|---|---|---|
| Table 2 likelihood AUC: chrono 0.67 / PLINDER 0.58 / seq-sim 0.61 | 0.6682 / 0.5834 / 0.6117 ✓ | `evaluation_results/checkpoints_model_run_2025-07-02*` |
| Table 1 sim-enrich 7.58 / 5.50, diversity 0.86 / 0.88, QED 0.60 / 0.56, LogP 0.70 / 0.45 | all ✓ to the digit | `evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol` |
| Vina score (Table 1) and AutoDock Vina AUC 0.62 (Table 2) | ✗ **irreproducible** | `evaluation_results/autodock_vina/` deleted; no `docking_summary.csv` survives |

The two runs are demonstrably different models — on the same chrono split, May gives likelihood AUC
0.6135 and EF 7.58 / diversity 0.86, while July gives 0.6682 and EF 6.32 / diversity 0.99.

The run name suggests July loaded from `checkpoints/` (the released HF weights) while `bubba_zjhnye4j`
is a W&B run id for an unreleased training run — **this is inferred, not verified**; neither directory
stores a config or log recording the checkpoint path. Testable: rerun the eval against `checkpoints/`
and see whether it lands on 0.5834 / 0.6682 / 0.6117.

**Before quoting any number in the thesis, regenerate both tables from one checkpoint.**

Reported AUC convention = **mean of per-system AUCs**, consistently for all three splits
(`calculate_autodock_vina_roc_auc.py` does exactly this; July-run means are 0.6682 / 0.5834 / 0.6117 vs
published 0.67 / 0.58 / 0.61). Per-system *medians* are much higher (0.739 / 0.623 / 0.686) — don't
confuse the two.

---

## 4b. W&B runs behind the released checkpoints

Entity `cath`; projects `poc2mol` and `voxelSmiles`. Credentials are in `~/.netrc`, so
`wandb.Api()` works locally. Local `wandb/` only holds Jan–Feb runs; everything else is server-side.

**Poc2Mol** — released `checkpoints/poc_vox_to_mol_vox/epoch_173.ckpt` (epoch 173, global_step 13572):

| | |
|---|---|
| Run | **`ljv96zyo`** — https://wandb.ai/cath/poc2mol/runs/ljv96zyo |
| Name / host | `poc2mol_HiQBind_kasp` on **kaspian**, started 2025-04-21T17:24:55Z, finished, 28.6 h |
| Reached | epoch 394 — the released ckpt is an **intermediate** checkpoint from this run |
| Args | `experiment=train_poc2mol_hiqbind data.num_workers=3 data.config.batch_size=2 +trainer.num_sanity_val_steps=0 trainer.val_check_interval=null model.lr=…` |
| Evidence | **Definitive** — its console log records `logs/poc2mol/runs/2025-04-21_18-13-26`, the exact path the combined model's config references |

**Combined** — released `checkpoints/combined_protein_to_smiles/epoch_000.ckpt` (epoch 0, global_step
**3,960,000**). Three-run resume chain, each link confirmed by the successor's recorded `ckpt_path`:

| # | Run | URL | Host / start | End global_step |
|---|---|---|---|---|
| 1 | **`crz11hbc`** | https://wandb.ai/cath/voxelSmiles/runs/crz11hbc | kaspian, 2025-03-22T21:19:22Z, finished, 116 h | 2,656,049 |
| 2 | **`55vlvc7x`** | https://wandb.ai/cath/voxelSmiles/runs/55vlvc7x | bubba-213-2, 2025-05-06T19:51:55Z, crashed, 67 h | 3,334,699 |
| 3 | **`zjhnye4j`** | https://wandb.ai/cath/voxelSmiles/runs/zjhnye4j | bubba-213-1, 2025-05-09T03:40:23Z, crashed, 92 h | 4,602,849 |

- The released step **3,960,000 falls inside run 3** (which spans 3.33 M → 4.60 M at epoch 0→1), so the
  released checkpoint is an **intermediate checkpoint of `zjhnye4j`**, not its final state.
- `zjhnye4j`'s `task_name` (`CombinedHiQBAggPropPoc2Mol`) and `ckpt_path` match the released
  `config.yaml` exactly. Its sibling **`7yzj4c06`** (`CombinedHiQBindHigherPropPoc2Mol`, ends 4,571,099)
  also resumed from `55vlvc7x` and also passes through 3.96 M — ruled out only by `task_name`.
- `crz11hbc` console log confirms dir `logs/vox2smilesZincAndPoc2MolOutputs/runs/2025-03-22_21-18-58`
  (the `_from_kaspian` suffix was added when the dir was copied to bubba).
- `55vlvc7x` and `zjhnye4j` have **no `output.log`** on the server (crashed bubba runs), so they're
  matched by timestamp + `task_name` + the `ckpt_path` chain, not by log. Timestamps convert as
  **BST = UTC+1** after 30 Mar 2025; March runs are UTC.

**This revises §4.** The two paper tables are *not* two unrelated models — they're most likely two
different checkpoints **of the same run `zjhnye4j`**: the May eval dir `bubba_zjhnye4j_2025-05-11_…`
evaluated a checkpoint taken ~May 11 while the run was still going (it ended ~May 12 23:30 UTC), whereas
the July eval used the released step-3,960,000 checkpoint. Still means Table 1 and Table 2 are not from
one checkpoint, but the gap is training steps within a run, not different models.

⚠️ **`crz11hbc` has `ckpt_path: None`** and was launched as `+experiment=train_vox2smiles_combined` (an
experiment config no longer in the repo). So the released decoder's traceable resume chain starts from
scratch at `crz11hbc`, and the separately-trained ZINC-only runs (`s7xnxqhu`, `lryunro2`, `t2gly4xp`) are
**not** in its lineage. The paper describes ZINC pretraining *then* fine-tuning on Poc2Mol grids; the
actual released model appears to have trained on the **mixture from the start** (that config used
`prob_poc2mol: 0.3`, so ~70% ZINC). Worth checking before repeating the paper's description.

### Re-training comparison baselines
Pull curves for `ljv96zyo` (Poc2Mol) and `zjhnye4j` (combined). Note both released checkpoints are
*intermediate*, so compare at matched `trainer/global_step`, not final values. Beware `_step` (W&B
logging counter) ≠ `trainer/global_step` — they differ by ~75× here.

---

## 5. Evaluation findings

### 5.1 The likelihood metric is dominated by a ligand-size nuisance term

`evaluate_combined_vox2smiles.py:352-394` writes one CSV per pocket under
`.../plixer_likelihood_scores/likelihood_scores/`. They store only `is_hit`, `likelihood` and the
*pocket* id — **no ligand identity** — but identity is fully recoverable because decoys are built as
`[s for s in df.smiles.values if s not in batch['smiles']]`, i.e. the test-split CSV in fixed order with
the pocket's own SMILES removed. Row 0 is the hit. Reconstruction verified exact (row counts match for
all 943 files including the 27 duplicate-SMILES systems; recomputed AUCs match the stored ones to 1e-16).
Script: see §7.

Variance decomposition of the 943×943 matrix:

```
pocket effect    6.4%
ligand effect   84.0%   <- intrinsic molecule likelihood, mostly SIZE
interaction      9.5%   <- the only part that can encode pocket specificity
```

Likelihood is mean per-token cross-entropy, so it tracks molecular size:
`corr(ligand mean likelihood, heavy-atom count) = -0.758`. For the ranking task this term is **pure
noise, not competing signal** — a pocket-blind baseline scoring by ligand mean alone gets AUC exactly
0.500. It just swamps the interaction term.

Removing it (z-score each ligand's column across pockets):

| Split | Raw (published) | Z-normalised |
|---|---|---|
| chrono | 0.67 | **0.854** |
| PLINDER | 0.58 | **0.753** |
| seq-sim | 0.61 | **0.782** |

Controls all pass (computed on the May run, where the full 943×943 matrix was reconstructed first):
the column view (fix ligand, rank pockets) independently gives 0.8493 vs the z-normalised 0.8497;
size-matched decoys (±10% heavy atoms) still 0.804; excluding same-protein sister pockets 0.8500;
permutation null 0.543; pocket-blind baseline exactly 0.500. No leakage — `forward` calls the same
`compute_smiles_metrics` on the same `predicted_ligand_voxels` used for decoys, and Poc2Mol only ever
sees protein voxels. Robust across both checkpoints (0.850 May / 0.854 July), so the conclusion does
not depend on which one you settle on.

**Implication:** on PLINDER this is 0.753 vs AutoDock Vina's 0.62. The paper's claim that Plixer
likelihoods are *"slightly less effective in ranking than docking scores"* is an artifact of the
normalisation. **Deployable prospectively** two ways: (a) score against a fixed background panel of
~100 reference pockets and z-score; (b) better — use the **likelihood ratio**
`log P(S | pocket) − log P(S | pocket-free reference)` with the ZINC-only decoder as reference. That's
the pointwise mutual information; the current metric is the numerator alone.

### 5.2 Data splits — protein-level clean, ligand-level leaky

Verified from the released checkpoints' own configs:
- **Poc2Mol**: `../hiqbind/parquet/{train,val,test}`. Test = chronological (2020+). Val = carved from the
  **pre-2020 pool by protein cluster**, *not* chronological — and cleanly cluster-disjoint from train.
- **Combined**: ZINC + Poc2Mol outputs over `../hiqbind/parquet/**train**` (correct split).
- **Vox2Smiles**: `../zinc20_parquet`, no protein split (ligand-only; deliberate).

| Check | Result |
|---|---|
| train ∩ val ∩ test system_ids | **0** everywhere |
| train/val protein clusters | **0 shared** — 10% cluster holdout worked |
| train ∩ test protein clusters | 191 — expected for a chronological split; this is why the strict subsets exist |
| 943 chrono / 107 PLINDER / 141 seq-sim eval systems | **all in parquet test, none in train** |

So no evaluation *pocket* was ever trained on. But **no ligand constraint was applied**:

- 21.3% chrono / 11.2% PLINDER / 14.2% seq-sim test ligands appear verbatim in Poc2Mol/combined **train**
  (paired with a different pocket).
- 24% chrono / 30% PLINDER / 37% seq-sim appear verbatim in **ZINC20**, seen by the decoder.

Measured impact (stratified by whether the true ligand was ever seen):

| Metric | Seen | Unseen | Published |
|---|---|---|---|
| chrono sim-enrichment | EF 10.29 | EF 6.08 | 7.58 |
| PLINDER sim-enrichment | EF 6.86 | EF 4.72 | 5.50 |
| chrono raw AUC | 0.758 | 0.618 | 0.67 |
| seq-sim raw AUC | 0.696 | **0.552** | 0.61 |
| chrono z-norm AUC | 0.880 | 0.839 | — |
| seq-sim z-norm AUC | 0.799 | 0.771 | — |

On genuinely unseen ligands the raw seq-sim AUC falls to near chance (0.552). The z-normalised metric is
**3–5× less sensitive** to this contamination — an independent argument for it.

Call the strict subsets **"protein-novel"**, not "non-redundant": they control protein redundancy only.
Report headline metrics stratified by ligand novelty rather than re-splitting and retraining.

⚠️ Leakage figures are **lower bounds** — matched as exact SMILES strings without re-canonicalising.

### 5.3 Known weaknesses of the published evaluation
- Vina correlates with heavy-atom count; report **ligand efficiency** (Vina/HAC) — Pocket2Mol beats Plixer
  on QED/LogP, so "better Vina" may partly mean "bigger molecules".
- The 0.3 Morgan-Tanimoto hit threshold is arbitrary and below the level that implies shared activity.
- Global ROC-AUC is the wrong VS summary — use **EF@1%, EF@5%, BEDROC (α=20)**.
- Fig. 3 shows a spike at similarity 1.0 (exact recoveries) — audit against memorisation.

---

## 6. Available but unused evaluation data

Already on disk, would replace the arbitrary similarity proxy with real actives/inactives:

- `../LIT-PCBA_AVE_UNBIASED` — 15 targets, AVE-debiased, dose-response actives/inactives. **Best primary
  choice.** A started script exists: `evaluations/evaluate_combi_model_lit_pcba.py`.
- `../DUDe_binding_and_decoys` — 102 targets, property-matched decoys. Gap vs LIT-PCBA is diagnostic of
  property shortcuts.
- `../BindingDB/BindingDB_All.tsv` — replaces "the one PDB ligand" with *all* known actives per target.

Also worth adding: **PoseBusters** on docked generated molecules; **Boltz-2** affinity as a much better
oracle than Vina (`.boltz` already present on the node — but it's a model, not ground truth).

---

## 7. Analysis scripts

Now in the repo under `scripts/adhoc_analysis/` (prefix `paper_audit_`). Paths inside them are
absolute to the **local** `/mnt/disk2/VoxelDiffOuter/...` machine — they need repointing to
`~/plixer_outer/...` to run on the node, and they read `evaluation_results/`, which is gitignored
and currently **local-only** (48 GB, not copied to the node).

| Script | Does |
|---|---|
| `paper_audit_rebuild_matrix.py` | reconstructs the 943×943 pocket×ligand likelihood matrix (§5.1) |
| `paper_audit_analyse_matrix.py` | variance decomposition, z-normalisation, bootstrap CIs |
| `paper_audit_validate.py` | cross-check vs stored AUCs, size control, permutation null |
| `paper_audit_paper_run.py` | reproduces Table 2 from the July run + z-normalised values |
| `paper_audit_split_audit.py` | train/val/test overlap + ZINC leakage scan |
| `paper_audit_leak_impact.py` | stratifies metrics by ligand-seen-in-training |
| `paper_audit_wandb_runs.py` / `_lineage.py` / `_confirm.py` | W&B run inventory + lineage (§4b) |

## 7b. Provenance system (added 2026-08-06, uncommitted at time of writing)

`src/utils/provenance.py` + `ProvenanceCallback`, wired into `src/train.py` and mirrored into W&B
hparams by `logging_utils.py`; `scripts/describe_checkpoint.py` reads it back. Records git commit /
branch / dirty state, resolved config, parent checkpoints and hostname, **embedded into every
checkpoint**. This exists precisely because §4b was so painful to reconstruct — future runs should be
identifiable from a stray `.ckpt` alone. Use it for all re-training runs.

## 7c. Manuscript

`plixer-manuscript/` is a **separate git repo** nested in the working dir
(`git@github.com:JudeWells/plixer-manuscript.git`, local branch `phd_thesis_ablation` = origin/main
@ d397015). It is not tracked by the plixer repo and not in `.gitignore` — adding it would create an
embedded-repo warning. Leave it untracked or gitignore it.

## 8. Open items

1. Rerun eval against `checkpoints/` to confirm the released weights are the July model
   (§4b suggests both tables trace to run `zjhnye4j` at different steps — verify).
2. Regenerate Table 1 + Table 2 from a **single** checkpoint.
3. Decide what replaces Vina (needs re-docking from scratch either way).
4. Implement the likelihood-ratio scorer; check it reproduces the z-normalised numbers.
5. Redo leakage matching with canonical SMILES + Murcko scaffolds for a true figure.
6. Test DDP for the combined stage before any multi-GPU run.
