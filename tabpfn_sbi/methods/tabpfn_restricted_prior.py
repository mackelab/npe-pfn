from functools import partial

import torch
from sbi.utils import RestrictedPrior
from tabpfn import TabPFNClassifier


class TabPFNRestrictedPrior(RestrictedPrior):
    def __init__(self, prior, acceptance_threshold=0.3, tabpfn_classifier_kwargs=None):
        self.classifier = TabPFNClassifier(
            **tabpfn_classifier_kwargs if tabpfn_classifier_kwargs else {}
        )
        self.thetas = torch.tensor([])
        self.y_labels = torch.tensor([])
        # Pickable
        accept_reject_fn = partial(
            self.accept_reject_fn,
            classifier=self.classifier,
            acceptance_threshold=acceptance_threshold,
        )

        super().__init__(prior, accept_reject_fn)

    @staticmethod
    def accept_reject_fn(theta, classifier, acceptance_threshold=0.3):
        return torch.tensor(
            classifier.predict_proba(theta)[:, -1] > acceptance_threshold
        )

    def log_prob(
        self,
        theta,
        norm_restricted_prior=False,  # We set this to false for efficiency
        track_gradients=False,
        prior_acceptance_params=None,
    ):
        return super().log_prob(
            theta, norm_restricted_prior, track_gradients, prior_acceptance_params
        )

    def append_simulations(self, theta, y_label):
        theta = torch.tensor(theta).detach().cpu()
        y_label = torch.tensor(y_label).detach().cpu()
        self.thetas = torch.cat([self.thetas, theta], dim=0)
        self.y_labels = torch.cat([self.y_labels, y_label], dim=0)

        thetas_class0 = self.thetas[self.y_labels == 0]
        thetas_class1 = self.thetas[self.y_labels == 1]

        N_class0 = thetas_class0.shape[0]
        N_class1 = thetas_class1.shape[0]

        N_max = 10_000
        N_half = N_max // 2

        # Initialize variables
        _thetas_class0 = thetas_class0
        _thetas_class1 = thetas_class1

        # Sample class 0 if needed
        if N_class0 > N_half:
            idx_class0 = torch.randperm(N_class0)[:N_half]
            _thetas_class0 = thetas_class0[idx_class0]

        # Sample class 1 if needed
        if N_class1 > N_half:
            idx_class1 = torch.randperm(N_class1)[:N_half]
            _thetas_class1 = thetas_class1[idx_class1]

        # Handle imbalanced classes
        if N_class1 < N_half and N_class0 > N_half:
            additional_class0_samples = N_half - N_class1
            if N_class0 > additional_class0_samples:
                idx_class0 = torch.randperm(N_class0)[:additional_class0_samples]
                _thetas_class0 = torch.cat(
                    [_thetas_class0, thetas_class0[idx_class0]], dim=0
                )

        if N_class0 < N_half and N_class1 > N_half:
            additional_class1_samples = N_half - N_class0
            if N_class1 > additional_class1_samples:
                idx_class1 = torch.randperm(N_class1)[:additional_class1_samples]
                _thetas_class1 = torch.cat(
                    [_thetas_class1, thetas_class1[idx_class1]], dim=0
                )

        # Fit the classifier with the sampled data
        self.classifier.fit(
            torch.cat([_thetas_class0, _thetas_class1], dim=0),
            torch.cat(
                [
                    torch.zeros(_thetas_class0.shape[0]),
                    torch.ones(_thetas_class1.shape[0]),
                ],
                dim=0,
            ),
        )
