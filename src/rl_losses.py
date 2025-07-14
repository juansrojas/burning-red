import torch


class RLLosses:
    @staticmethod
    def mse_loss(estimate, target):
        return torch.nn.functional.mse_loss(estimate, target)

    @staticmethod
    def ac_policy_loss(log_prob, delta):
        return (-1 * log_prob * delta).mean()
