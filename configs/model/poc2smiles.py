import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from src.evaluation.visual import visualise_batch
from src.models.poc2mol import Poc2Mol
from ../voxmiles/src/models/vox2smiles import VoxToSmilesModel


class CombinedProteinToSmilesModel(LightningModule):
    def __init__(self, poc2mol_model, vox2smiles_model, config):
        super().__init__()
        self.poc2mol = poc2mol_model
        self.vox2smiles = vox2smiles_model
        self.config = config

    def forward(self, protein_voxel):
        # Generate molecule voxel from protein
        mol_voxel, _ = self.poc2mol(protein_voxel)

        # Generate SMILES from molecule voxel
        smiles_output = self.vox2smiles(mol_voxel)

        return mol_voxel, smiles_output

    def training_step(self, batch, batch_idx):
        protein_voxel = batch["protein"]
        target_mol_voxel = batch["ligand"]
        target_smiles = batch["smiles"]

        pred_mol_voxel, pred_smiles = self(protein_voxel)

        # Calculate losses
        mol_loss = self.poc2mol.model.loss(pred_mol_voxel, target_mol_voxel)
        smiles_loss = self.vox2smiles.criterion(pred_smiles.logits, target_smiles)

        total_loss = mol_loss + smiles_loss

        self.train_loss(total_loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        protein_voxel = batch["protein"]
        target_mol_voxel = batch["ligand"]
        target_smiles = batch["smiles"]

        pred_mol_voxel, pred_smiles = self(protein_voxel)

        # Calculate losses
        mol_loss = self.poc2mol.model.loss(pred_mol_voxel, target_mol_voxel)
        smiles_loss = self.vox2smiles.criterion(pred_smiles.logits, target_smiles)

        total_loss = mol_loss + smiles_loss

        self.val_loss(total_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Visualize results (adapt as needed)
        if batch_idx < 3:
            save_dir = f"{self.config.img_save_dir}/val"
            visualise_batch(
                target_mol_voxel[:4],
                pred_mol_voxel[:4],
                batch["name"][:4],
                save_dir=save_dir,
                batch=str(batch_idx)
            )
            self.vox2smiles.visualize_smiles(batch, pred_smiles)

        return total_loss

    def test_step(self, batch, batch_idx):
        # Similar to validation_step, but with additional metrics if needed
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

if __name__=="__main__":
    # instantiate Poc2Mol and Vox2Smiles models
    poc2mol = Poc2Mol()


    pass