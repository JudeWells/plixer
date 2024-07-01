from lightning import LightningDataModule
from torch.utils.data import DataLoader

class VoxMilesDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        vox_size: float,
        box_dims: list,
        max_smiles_len: int = 256,
        batch_size: int = 32,
        num_workers: int = 0,
        random_rotation: bool = True,
        random_translation: float = 6.0,
    ):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_datset)
