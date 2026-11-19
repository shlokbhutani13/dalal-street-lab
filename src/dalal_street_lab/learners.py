"""The models we compare. Each one learns from past features and guesses tomorrow's direction.

Every learner has the same two methods, so the testing code can treat them all alike
(this "same shape" idea is a Protocol, like a plug that fits one socket):

    fit(x, y)            learn from training rows
    predict_proba(x)     for each row, the chance that tomorrow is an up day (0 to 1)

Two of them are deliberately dumb baselines. A real model has to beat them, or it
isn't learning anything useful.
"""

from typing import Protocol

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

FloatArray = NDArray[np.float64]


class Learner(Protocol):
    name: str

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_]) -> None: ...

    def predict_proba(self, x: pd.DataFrame) -> FloatArray: ...


class AlwaysUp:
    """Baseline 1: always says "up", with the confidence of how often training days went up.

    Markets rise a bit more often than they fall, so this is harder to beat than it sounds.
    """

    name = "always_up"

    def __init__(self) -> None:
        self.up_rate = 0.5

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_]) -> None:
        self.up_rate = float(np.mean(y))

    def predict_proba(self, x: pd.DataFrame) -> FloatArray:
        return np.full(len(x), self.up_rate)


class Momentum:
    """Baseline 2: "what went up this week keeps going up". No learning at all.

    The 5-day move is divided by normal daily noise, so a big move in a calm stock counts
    for more than the same move in a jumpy one. A sigmoid squashes it into 0 to 1.
    """

    name = "momentum"

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_]) -> None:
        return None

    def predict_proba(self, x: pd.DataFrame) -> FloatArray:
        noise = x["volatility_20d"].to_numpy() * np.sqrt(5) + 1e-9
        strength = x["ret_5d"].to_numpy() / noise
        return np.asarray(1 / (1 + np.exp(-strength)), dtype=np.float64)


class Logistic:
    """Logistic regression: a weighted sum of the features, squashed into a probability.

    The scaler first puts every feature on the same scale (mean 0, spread 1), so RSI
    (0 to 100) doesn't drown out returns (around 0.01).
    """

    name = "logistic"

    def __init__(self, features: list[str], c: float = 0.1) -> None:
        self.features = features
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1_000))

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_]) -> None:
        self.model.fit(x[self.features].to_numpy(), y)

    def predict_proba(self, x: pd.DataFrame) -> FloatArray:
        return np.asarray(self.model.predict_proba(x[self.features].to_numpy())[:, 1])


class _Mlp(nn.Module):
    def __init__(self, inputs: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.layers(x)
        return logits.squeeze(-1)  # one number per row: the "up" score (a logit)


class NeuralNet:
    """A small PyTorch neural network (12 -> 32 -> 16 -> 1).

    Early stopping: the newest 15% of the training rows are held back. Training stops when
    the model stops improving on them, before it memorises noise. Held-back rows are the
    NEWEST ones on purpose, because that's what the future looks like to the model.
    """

    name = "neural_net"

    def __init__(
        self,
        features: list[str],
        *,
        seed: int = 7,
        max_epochs: int = 200,
        patience: int = 10,
        dropout: float = 0.2,
    ) -> None:
        self.features = features
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.dropout = dropout
        self.mean = np.zeros(len(features))
        self.std = np.ones(len(features))
        self.net: _Mlp | None = None

    def _tensor(self, x: pd.DataFrame) -> torch.Tensor:
        values = (x[self.features].to_numpy(dtype=np.float64) - self.mean) / self.std
        return torch.tensor(values, dtype=torch.float32)

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_]) -> None:
        torch.manual_seed(self.seed)  # same data in, same model out
        self.mean = x[self.features].to_numpy(dtype=np.float64).mean(axis=0)
        self.std = x[self.features].to_numpy(dtype=np.float64).std(axis=0) + 1e-9

        split = int(len(x) * 0.85)  # rows are in time order: the last 15% are the newest
        inputs, targets = self._tensor(x), torch.tensor(y, dtype=torch.float32)
        train_x, train_y = inputs[:split], targets[:split]
        check_x, check_y = inputs[split:], targets[split:]

        net = _Mlp(len(self.features), self.dropout)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        best_loss, best_state, waited = float("inf"), net.state_dict(), 0
        generator = torch.Generator().manual_seed(self.seed)

        for _ in range(self.max_epochs):
            net.train()
            order = torch.randperm(len(train_x), generator=generator)
            for batch in torch.split(order, 256):
                optimizer.zero_grad()
                loss_fn(net(train_x[batch]), train_y[batch]).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                check_loss = float(loss_fn(net(check_x), check_y))
            if check_loss < best_loss - 1e-5:
                best_loss, waited = check_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        net.load_state_dict(best_state)
        self.net = net

    def predict_proba(self, x: pd.DataFrame) -> FloatArray:
        if self.net is None:
            raise RuntimeError("Call fit() before predict_proba().")
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(self._tensor(x))).numpy().astype(np.float64)


def default_learners(features: list[str]) -> list[Learner]:
    return [AlwaysUp(), Momentum(), Logistic(features), NeuralNet(features)]
