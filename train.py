import torch
import torch.nn as nn
import argparse
from einops import rearrange
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchsummary as summary
import torch.optim as optim
import os
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt

class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        #nn.SiLU()
        SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        #nn.SiLU()
        SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            #nn.SiLU(),
            SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        #print(99999999)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.SiLU(),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.SiLU(),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.SiLU(),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):#x==tensor
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        _, _, h, w = x.shape
        #if x.numel() == 147456:
        #    _, _, h, w = [1, 144, 32, 32]
        #elif x.numel() == 49152:
        #    _, _, h, w = [1, 192, 16, 16]
        #elif x.numel() == 15360:
        #    _, _, h, w = [1, 240,  8,  8]

        #print(x.shape)
        #print(x.numel())
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)#ph 2 pw 2
        #print(x.shape)
        x = self.transformer(x)
        #print("第二次rearrange：")
        #print(x.shape)
        #print(type(x))
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)#
        #print(x.shape)
        #print("第二次rearrange完成")
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)
    
        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_res(model,train_dataloader,epoch):
    #print("train_res")
    model.train()
    #print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    #print("len_dataset:", (len(train_datasets) ))
    #i = 1
    for batch_x, batch_y in train_dataloader:
        #print("batch_x:", batch_x[0][0][0])
        #print(batch_x[0].shape)
        #print("batch_y:", batch_y)
        batch_x  = Variable(batch_x).cuda()
        batch_y  = Variable(batch_y).cuda()
        optimizer.zero_grad()
        out = model(batch_x)
        #print(out)
        loss1 = loss_func(out, batch_y)
        train_loss += loss1.item()
        pred = torch.max(out, 1)[1]
        #print(pred)
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        loss1.backward()
        optimizer.step()
        #print("dataset: " + str(i) + "/" + str(len(train_datasets)/batch_size) + str('   {:.2f}   '.format(i/len(train_datasets)*batch_size)) + str(epoch) + "  "+ str(EPOCH))
        #i = i+1
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_datasets)), train_acc / (len(train_datasets))))#输出训练时的loss和acc
    train_Loss_list.append(train_loss / (len(val_datasets)))
    train_Accuracy_list.append(100 * train_acc / (len(val_datasets)))

# evaluation--------------------------------
def val(model,val_dataloader):
    model.eval()
    eval_loss= 0.
    eval_acc = 0.
    for batch_x, batch_y in val_dataloader:
        batch_x = Variable(batch_x, volatile=True).cuda()
        batch_y = Variable(batch_y, volatile=True).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_datasets)), eval_acc / (len(val_datasets))))#输出测试时的loss和acc
    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))
        
# 保存模型的参数
#torch.save(model.state_dict(), 'ResNet18.pth')
#state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
#torch.save(state, 'ResNet18.pth')

def draw_test(x1, y1, x2, y2):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1,'-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2,'-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig('figs/test.png', dpi=300)
    #plt.show()

def draw_train(x3,y3,x4,y4):
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(x3, y3,'-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x4, y4,'-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.savefig('figs/train.png', dpi=300)
    #plt.show()


    

#model = mobilevit_s().eval()
model = mobilevit_xs()
#out = vit(img)


#train  
EPOCH = 300 
batch_size= 1
classes_num=1000
learning_rate=1e-3

'''定义Transform'''
 #对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(256),		#对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),		#水平翻转
    transforms.ToTensor(),					#转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	#进行归一化
])
#对测试集做变换
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dir = "./mini_imagenet/train"           #训练集路径
#定义数据集
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
#加载数据集
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True,num_workers=1,pin_memory=True)#,num_workers=16,pin_memory=False

val_dir = "./mini_imagenet/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True,num_workers=1,pin_memory=True)#,num_workers=16,pin_memory=True



summary.summary(model, input_size=(3,256,256),device="cpu")#我们选择图形的出入尺寸为(3,224,224)

#params = [{'params': md.parameters()} for md in model.children()
#          if md in [model.classifier]]
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=1e-4)
StepLR    = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)#按训练批次调整学习率，每30个epoch调整一次
loss_func = nn.CrossEntropyLoss()
#存储测试loss和acc
Loss_list = []
Accuracy_list = []
#存储训练loss和acc
train_Loss_list = []
train_Accuracy_list = []
#这俩作用是为了提前开辟一个
loss = []
loss1 = []



#log_dir = './log/mobilevit.pth'
log_dir = './log/mobilevit_1bz_xs.pth'
#def main():
if torch.cuda.is_available():
    model.cuda()
    print("using_cuda")
test_flag = False
# 如果test_flag=True,则加载已保存的模型
if test_flag:
    # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    val(model, val_dataloader)
    #如果只评估模型，则保留这个return，如果是从某个阶段开始继续训练模型，则去掉这个模型
    #同时把上面的False改成True
#    return

# 如果有保存的模型，则加载模型，并在其基础上继续训练
if os.path.exists(log_dir) and test_flag:
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('加载 epoch {} 成功！'.format(start_epoch))
else:
    start_epoch = 1
    print('无保存模型，将从头开始训练！')

for epoch in range(start_epoch, EPOCH):
    since = time.time()
    print('epoch {}'.format(epoch))#显示每次训练次数
    train_res(model, train_dataloader, epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 输出训练和测试的时间
    #通过一个if语句判断，让模型每十次评估一次模型并且保存一次模型参数
    epoch_num = epoch/10
    epoch_numcl = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0]
    print('epoch_num',epoch_num)
    if epoch_num in epoch_numcl:
        print('评估模型')
        val(model, val_dataloader)
        print('保存模型')
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, log_dir)
    
y1 = Accuracy_list
y2 = Loss_list
y3 = train_Accuracy_list
y4 = train_Loss_list

x1 = range(len(Accuracy_list))
x2 = range(len(Loss_list))
x3 = range(len(train_Accuracy_list))
x4 = range(len(train_Loss_list))

print(x1)
print(x2)
print(x3)
print(x4)

draw_test(x1,y1,x2,y2)
draw_train(x3,y3,x4,y4)
