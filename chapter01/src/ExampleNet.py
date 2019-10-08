
import torch
import torch.nn as nn

class ExampleNet(nn.Module):

	def __init__(self):
		super(ExampleNet, self).__init__()
		self.linear_3 = nn.Linear(in_features = 28*28, out_features = 128, bias = True)
		self.reLU_4 = nn.ReLU(inplace = False)
		self.linear_6 = nn.Linear(in_features = 128, out_features = 256, bias = True)
		self.reLU_8 = nn.ReLU(inplace = False)
		self.linear_7 = nn.Linear(in_features = 256, out_features = 128, bias = True)
		self.reLU_9 = nn.ReLU(inplace = False)
		self.linear_10 = nn.Linear(in_features = 128, out_features = 10, bias = True)

	def forward(self, x_para_1):
		x_reshape_5 = torch.reshape(x_para_1,shape = (-1,28*28))
		x_linear_3 = self.linear_3(x_reshape_5)
		x_reLU_4 = self.reLU_4(x_linear_3)
		x_linear_6 = self.linear_6(x_reLU_4)
		x_reLU_8 = self.reLU_8(x_linear_6)
		x_linear_7 = self.linear_7(x_reLU_8)
		x_reLU_9 = self.reLU_9(x_linear_7)
		x_linear_10 = self.linear_10(x_reLU_9)
		return x_linear_10
