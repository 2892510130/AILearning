import torch
import torch.nn.functional as F

def bpr_loss(pos: torch.Tensor, neg: torch.Tensor)-> torch.Tensor:
    """Bayesian Personalized Ranking Loss

    Parameters
    ----------
    pos : torch.Tensor
        Ranking logit (0..1)
    neg : torch.Tensor
        Ranking logit (0..1)
    
    Return
    ------
    loss scalar
    """
    diff = pos - neg
    return -F.logsigmoid(diff).mean()

def hinge_loss(pos: torch.Tensor, neg: torch.Tensor, margin:float)-> torch.Tensor:
    diff = pos - neg
    return torch.maximum(margin-diff, 0).mean()