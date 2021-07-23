import torch
import torch.nn as nn

#print(torch.cuda.get_device_name(0))

def double_conv(input_ch, output_ch):
        conv = nn.Sequential(
        nn.Conv2d(input_ch, output_ch, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_ch, output_ch, kernel_size=3),
        nn.ReLU(inplace=True))
        return conv

def crop_image(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(1,64)
        self.down_conv2 = double_conv(64,128)
        self.down_conv3 = double_conv(128,256)
        self.down_conv4 = double_conv(256,512)
        self.down_conv5 = double_conv(512,1024)
        
        self.uptrans1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(1024, 512)
        
        self.uptrans2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(512, 256)
        
        self.uptrans3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(256, 128)
        
        self.uptrans4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        
    def forward(self, image):
        #Encoder part
        x1 = self.down_conv1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)
        
        #Decoder part
        x = self.uptrans1(x9)
        y = crop_image(x7,x)
        x = self.up_conv1(torch.cat([x, y], 1))
        
        x = self.uptrans2(x)
        y = crop_image(x5,x)
        x = self.up_conv2(torch.cat([x, y], 1))
        
        x = self.uptrans3(x)
        y = crop_image(x3,x)
        x = self.up_conv3(torch.cat([x, y], 1))
        
        x = self.uptrans4(x)
        y = crop_image(x1,x)
        x = self.up_conv4(torch.cat([x, y], 1))
        
        x = self.out(x)
        # print(x.size())
        return x

if __name__ == "__main__":
    image = torch.rand(1, 1, 572, 572)
    model = UNet()
    print(model(image))
