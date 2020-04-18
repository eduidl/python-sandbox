import argparse
from pathlib import Path

import optuna
from optuna.pruners import SuccessiveHalvingPruner
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 128
DIR = Path(__file__).parent.resolve()


class Net(pl.LightningModule):

    def __init__(self, trial):
        super().__init__()
        self.trial = trial
        self.layers = []
        self.dropouts = []

        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout = trial.suggest_uniform('dropout', 0.2, 0.5)

        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = int(trial.suggest_loguniform(f'n_units_l{i}', 4, 128))
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, 10))

        for idx, layer in enumerate(self.layers):
            setattr(self, f'fc{idx}', layer)

        for idx, dropout in enumerate(self.dropouts):
            setattr(self, f'drop{idx}', dropout)

    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)

    def training_step(self, batch, batch_nb):
        data, labels = batch
        output = self.forward(data)
        return {'loss': F.nll_loss(output, labels)}

    def validation_step(self, batch, batch_nb):
        data, labels = batch
        output = self.forward(data)
        predicted = output.argmax(dim=1)
        return {'val_loss': F.nll_loss(output, labels), 'val_accuracy': predicted.eq(labels).float().mean()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def on_post_performance_check(self):
        accuracy = self.trainer.callback_metrics['val_accuracy']

        self.trial.report(accuracy, self.current_epoch)
        if self.trial.should_prune(self.current_epoch):
            raise optuna.exceptions.TrialPruned()

    def configure_optimizers(self):
        optimizer_name = self.trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        lr = self.trial.suggest_uniform('lr', 1e-5, 1e-1)
        return getattr(optim, optimizer_name)(self.parameters(), lr=lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
                          shuffle=True,
                          batch_size=BATCH_SIZE,
                          num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(datasets.MNIST(DIR, train=False, download=True, transform=transforms.ToTensor()),
                          batch_size=BATCH_SIZE,
                          num_workers=4)


def objective_wrapper(epochs: int):

    def objective(trial):
        trainer = pl.Trainer(max_nb_epochs=epochs)
        model = Net(trial)
        trainer.fit(model)
        return trainer.callback_metrics['val_accuracy']

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize', pruner=SuccessiveHalvingPruner())
    study.optimize(objective_wrapper(args.epochs), n_trials=args.trials)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    study.trials_dataframe().to_csv(DIR / 'result.csv')


if __name__ == '__main__':
    main()
