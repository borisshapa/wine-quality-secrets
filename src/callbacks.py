import tqdm
import wandb


class WAndBCallback:
    def __init__(self, iterations: int):
        self.iterations = iterations
        self.pbar = tqdm.tqdm(total=iterations)

    def after_iteration(self, info) -> bool:
        f1_values = info.metrics["validation"]["TotalF1:average=Micro"]
        train_losses = info.metrics["learn"]["MultiClass"]
        val_losses = info.metrics["validation"]["MultiClass"]
        wandb.log(
            {
                "val/f1 micro": f1_values[-1],
                "train/loss": train_losses[-1],
                "val/loss": val_losses[-1],
            }
        )
        self.pbar.update()
        if len(train_losses) == self.iterations:
            self.pbar.close()
        return True
