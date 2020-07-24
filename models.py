import qcnn
import torch.nn as nn
import torch.nn.functional as F

class QCNN(nn.Module):
    def cuda(self):
        qcnn.cuda()
        return super(QCNN, self).cuda()
        
    def __init__(self, outclasses):
        super(QCNN, self).__init__()
        
        self.qnorm1 = qcnn.QBatchNorm1d(1) # New layer
        
        self.qconv1 = qcnn.QConv1d(1, 16, 5) # 16 * 96
        self.qnorm1 = qcnn.QBatchNorm1d(16)

        # time = 96
        self.qconv2 = qcnn.QConv1d(16, 32, 7, stride=2)
        self.qnorm2 = qcnn.QBatchNorm1d(32)
        
        # time = 45
        # invariant
        
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.norm3 = nn.BatchNorm1d(32)
        
        # time = 41
        self.conv4 = nn.Conv1d(32, 32, 7)      
        self.norm4 = nn.BatchNorm1d(32)
        
        # Now 32 channels x 35 time
        
        self.dense5 = nn.Linear(32*35, 768)
        self.dense6 = nn.Linear(768, outclasses)
        
    
    def forward(self, x):
        x = self.qconv1(x)
        x = self.qnorm1(x)
        
        x = self.qconv2(x)
        x = self.qnorm2(x)
        
        x = qcnn.qnorm(x)
        
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        
        x = x.view(-1, 32*35)
        x = F.selu(self.dense5(x))
        
        logits = self.dense6(x)
        return F.softmax(logits, dim=1)    

class CNN(nn.Module):
        
    def __init__(self, outclasses):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 5) # 16 * 96
        self.norm1 = nn.BatchNorm1d(64)

        # time = 96
        self.conv2 = nn.Conv1d(64, 32, 7, stride=2)
        self.norm2 = nn.BatchNorm1d(32)
        
        # time = 45
        # invariant
        
        self.conv3 = nn.Conv1d(32, 32, 5)
        self.norm3 = nn.BatchNorm1d(32)
        
        # time = 41
        self.conv4 = nn.Conv1d(32, 32, 7)      
        self.norm4 = nn.BatchNorm1d(32)
        
        # Now 32 channels x 35 time
        
        self.dense5 = nn.Linear(32*35, 768)
        self.dense6 = nn.Linear(768, outclasses)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        
        x = x.view(-1, 32*35)
        x = F.selu(self.dense5(x))
        
        logits = self.dense6(x)
        return F.softmax(logits, dim=1)    
