from src.models.train import Trainer
import pytest

@pytest.mark.parametrize("test_input", [0.1, 0.2])
def test_eval(test_input):
    test_training(test_input)


def test_training(lr=0.1):

    trainer = Trainer()

    trainer.args.lr = lr
    trainer.args.n_epochs = 1
    trainer.args.run_name = "pytest"
    trainer.args.use_wandb = False
    trainer.args.plot_results = False

    train_losses, test_losses = trainer.train()

    for l1, l2 in zip(train_losses, test_losses):
        assert l1 > 0
        assert l2 > 0
