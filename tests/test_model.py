import torch
from src.models.predict_model import predict


def test_model_output():

    accuracy, log_ps, images = predict(model_path="models/model29.pth",
                                       input_images=None,
                                       shuffle=False)

    assert accuracy == 0.9829000234603882
    assert log_ps.shape == torch.Size([16, 10])
    assert images.shape == torch.Size([16, 1, 28, 28])
