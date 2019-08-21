import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_GAN():
	checkpoint_path = "saved/checkpoint_GAN_20.pth"
	saved_path = "saved/img/GAN_loss.png"

	ckpt = torch.load(checkpoint_path)
	log = ckpt['log']

	epoch = [item['epoch'] for item in log]
	g_loss = [item['Gen_loss'] for item in log]
	d_loss = [item['Dis_loss'] for item in log]

	result = np.where(g_loss == np.amin(g_loss))
	print(result)

	fig, (ax1, ax2) = plt.subplots(2,1)
	ax1.plot(epoch, g_loss)
	ax1.set_ylabel('G_LOSS')
	ax2.plot(epoch, d_loss)
	ax2.set_xlabel('EPOCH')
	ax2.set_ylabel("D_LOSS")

	fig.tight_layout()
	fig.savefig(saved_path)

def plot_UNET():
	checkpoint_path = "saved/checkpoint_UNET_100.pth"
	saved_path = "saved/img/UNET_loss.png"

	ckpt = torch.load(checkpoint_path)
	log = ckpt['log']

	epoch = [item['epoch'] for item in log]
	g_loss = [item['Gen_loss'] for item in log]

	result = np.where(g_loss == np.amin(g_loss))
	print(result)

	fig, ax1 = plt.subplots(1,1)
	ax1.plot(epoch, g_loss)
	ax1.set_ylabel('G_LOSS')
	ax1.set_xlabel('EPOCH')

	fig.tight_layout()
	fig.savefig(saved_path)

if __name__ == '__main__':
	plot_GAN()