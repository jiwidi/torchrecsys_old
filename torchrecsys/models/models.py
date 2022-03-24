import torch
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    
    def forward(self, x):
        raise NotImplementedError("`forward` method must be implemented by the user")

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = self.task(*x)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def compile(self,):
        pass