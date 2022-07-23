import torch
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn
from Generator import Generator
from Datasets import LoadDataset
from os.path import join
from weight_init import weight_init

save_root = './model_save/37/epoch30'
dir_root = '/home/dog/tbw/Dataset/muti'
h_im_dir = ''
l_im_dir = ''
color = 0
G_inchannel, G_outchannel = 3, 64
batch_size = 16
image_size = 64
num_epochs = 160
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
datasets = LoadDataset(dir_root, h_im_dir, l_im_dir, image_size=image_size, height=256, width=448, transform=transform)
dataloader = data.DataLoader(datasets, batch_size=batch_size, drop_last=True, shuffle=True)
G = Generator(G_inchannel, G_outchannel).to(device)
G.apply(weight_init)
G.load_state_dict(torch.load('./model_save/37/epoch30/G.pth', map_location=device))
evaluation = nn.MSELoss(reduction='mean').to(device)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.9, 0.999))
total_steps = len(dataloader)
G.train()
for epoch in range(num_epochs):
    accurate = total_steps
    for i, (data, target) in enumerate(dataloader):
        G.init_colorpanel(color, data)
        data = data.to(device)
        target = target.to(device)
        output = G(data)
        g_loss = evaluation(output, target)
        g_loss = g_loss / len(data[1:])
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if i % 10 == 0:
            torch.save(G.state_dict(), join(save_root, 'G.pth'))
        print("Epoch[{}/{}],Step[{}/{}],g_loss:{:.4f}".format(epoch, num_epochs, i, total_steps, g_loss.item()))
