import numpy as np
import torch
import torch.nn.functional as F

# Transforming the fbank features to match the receptive field and stride of the W2V2, HuBERT CNN layer outputs
PER_LAYER_TRANSFORM_DCT = {
    1: {"kernel": 79, "stride": 32},
    2: {"kernel": 39, "stride": 16},
    3: {"kernel": 19, "stride": 8},
    4: {"kernel": 9, "stride": 4},
    5: {"kernel": 4, "stride": 2},
    6: {"kernel": 2, "stride": 1},
    7: {"kernel": 1, "stride": 2},
    "avhubert0": {"kernel": 4, "stride": 4},
}


def transform_rep(kernel_size, stride, layer_rep):
    """
    Transform local z representations to match the fbank features' stride and receptive field

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_rep = torch.transpose(layer_rep, 1, 0)  # 512 x 1 x num_frames
    weight = (
        torch.from_numpy(np.ones([1, 1, kernel_size]) / kernel_size)
        .type(torch.cuda.FloatTensor)
        .to(device)
    )
    transformed_rep = F.conv1d(layer_rep, weight, stride=stride)
    transformed_rep = torch.transpose(transformed_rep, 1, 0)

    # check averaging
    mean_vec1 = torch.mean(layer_rep[:, :, :kernel_size], axis=-1)
    mean_vec2 = torch.mean(layer_rep[:, :, stride : stride + kernel_size], axis=-1)
    out_vec1 = transformed_rep[:, :, 0]
    out_vec2 = transformed_rep[:, :, 1]
    assert torch.mean(mean_vec1 - out_vec1) < 2e-8
    assert torch.mean(mean_vec2 - out_vec2) < 2e-8
    return torch.transpose(transformed_rep, 1, 2).squeeze(0).cpu().numpy()
