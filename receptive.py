import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_receptive_field import receptive_field
from model import GGGAN, GGGGAN, SNDIS, ResDense


if __name__ == '__main__':
	device = torch.device('cuda')
	model = GGGAN.Discriminator().to(device)

	if True:
		receptive_field(model, (2, 640, 480))
	else:
		receptive_field_dict = receptive_field(model, (2, 640, 480))
		receptive_field_for_unit(receptive_field_dict, "2", (2,2))
