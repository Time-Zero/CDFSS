import torch

from model.segformer import SegFormer
from model.IFA_matching import IFA_MatchingNet

if __name__ == '__main__':
    model_file_path = '/home/ymc/CDFSS/data/weights/segformer_b0_weights_voc.pth'
    segformer_model = SegFormer()
    segformer_model.load_state_dict(torch.load(model_file_path, weights_only=True), strict=False)

    ifa_model = IFA_MatchingNet(segformer_model)
