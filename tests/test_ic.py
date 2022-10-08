import math
import torch
from src.models.losses import iou, multibox_iou, multibox_generalized_iou, \
    multibox_distance_iou, multibox_complete_iou, multibox_full_iou

def test_iou():
    a = torch.tensor([0., 1., 2., 3.])
    b = torch.tensor([0., 1., 2., 3.])
    assert iou(a, b).equal(torch.tensor(1.))

    b = torch.tensor([2., 3., 4., 5.])
    assert iou(a, b).equal(torch.tensor(0.))

    b = torch.tensor([1., 2., 3., 4.])
    assert iou(a, b).equal(torch.tensor(1/7))

def test_multibox_iou():
    t_ = torch.tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    u_ = torch.tensor([[2., 3., 4., 5.], [1., 2., 3., 4.]])

    assert multibox_iou(t_, u_).equal(torch.tensor((1 + 1/7) / 2))

def test_multibox_giou():
    t_ = torch.tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    u_ = torch.tensor([[2., 3., 4., 5.], [1., 2., 3., 4.]])

    assert multibox_generalized_iou(t_, u_).equal(torch.tensor(1 + 1/7 - 2/9) / 2)

def test_multibox_diou():
    t_ = torch.tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    u_ = torch.tensor([[2., 3., 4., 5.], [1., 2., 3., 4.]])

    assert multibox_distance_iou(t_, u_).equal(torch.tensor(1 + 1/7 - 1/9)/2)

def test_multibox_ciou():
    t_ = torch.tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    u_ = torch.tensor([[2., 3., 4., 5.], [1., 2.5, 3., 3.5]])

    v = (2 * (torch.atan(torch.tensor(2.)) - torch.atan(torch.tensor(1.))) / math.pi) ** 2
    assert multibox_complete_iou(t_, u_).equal(
        (1 + 1/11 - 8 / 61 - (v ** 2 / (1 + v - 1/11 + 1e-7)))/2)

def test_multibox_fiou():
    t_ = torch.tensor([[0., 1., 2., 3.], [2., 3., 4., 5.]])
    u_ = torch.tensor([[2., 3., 4., 5.], [1., 2.5, 3., 3.5]])

    v = (2 * (torch.atan(torch.tensor(2.)) - torch.atan(torch.tensor(1.))) / math.pi) ** 2
    assert multibox_full_iou(t_, u_).equal(
        (1 + 1/11 - 4 / 15 - 8 / 61 - (v ** 2 / (1 + v - 1/11 + 1e-7)))/2)
