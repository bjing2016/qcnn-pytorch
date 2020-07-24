import torch
import torch.nn as nn
from torch.autograd import Function
import math


Qmt = torch.Tensor([[
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
],[
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, -1],
    [0, 0, 1, 0]
],[
    [0, 0, -1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, -1, 0, 0]
],[
    [0, 0, 0, -1],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
]]).float()


Qmt2 = torch.Tensor([[
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
],[
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 0]
],[
    [0, 0, -1, 0],
    [0, 0, 0, -1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
],[
    [0, 0, 0, -1],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0]
]]).float()

Qmt3 = torch.Tensor([[
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1]
],[
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 0]
],[
    [0, 0, 1, 0],
    [0, 0, 0, -1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
],[
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0]
]]).float()

transposer = torch.Tensor([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1]
]).float()

def cuda():
    global Qmt
    Qmt = Qmt.cuda()
    global Qmt2
    Qmt2 = Qmt.cuda()
    global Qmt3
    Qmt3 = Qmt3.cuda()
    global transposer
    transposer = transposer.cuda()
    
def q2m(qs, M):
    '''
    |qs|: dims (*, 4, 1)
    |m|: dims (4, 4, 4)
    '''
    qs = qs.unsqueeze(-1) # dims (*, 4, 1, 1)
    qs = M * qs # dims (*, 4, 4, 4)
    return qs.sum(-3) # dims (*, 4, 4)

def qconj(q):
    return transposer.matmul(q)

def qnormsq(q):
    return (q**2).sum(-2, keepdim=True) # 30x speed

def qinv(q):
    return qconj(q)/qnormsq(q) # 20x speed
    
class QMultiply(Function):
    @staticmethod
    def forward(ctx, p, q):
        ctx.save_for_backward(p, q)
        return q2m(p, Qmt).matmul(q)
        
    @staticmethod
    def backward(ctx, grad_output):
        p, q = ctx.saved_tensors
        dout_dp = q2m(q, Qmt2)
        dout_dq = q2m(p, Qmt)
        grad_p = dout_dp.transpose(-1, -2).matmul(grad_output)
        grad_q = dout_dq.transpose(-1, -2).matmul(grad_output)
        return grad_p, grad_q


class QMultiplyConjugate(Function):
    @staticmethod
    def forward(ctx, p, q):
        ctx.save_for_backward(p, q)
        return q2m(p, Qmt).matmul(qconj(q))
        
    @staticmethod
    def backward(ctx, grad_output):
        p, q = ctx.saved_tensors
        dout_dp = q2m(q, Qmt2)
        dout_dq = q2m(p, Qmt3)
        grad_p = dout_dp.matmul(grad_output)
        grad_q = dout_dq.transpose(-1, -2).matmul(grad_output)
        return grad_p, grad_q

class QConjugate(Function):
    @staticmethod
    def forward(ctx, p, q): # qpq*/||q||^2
        ctx.save_for_backward(p, q)
        return QMultiply.apply(q, QMultiplyConjugate.apply(p, q))/qnormsq(q)
    
    @staticmethod
    def backward(ctx, grad_output):
        p, q = ctx.saved_tensors
        qp = QMultiply.apply(q, p)
        qpq = QMultiplyConjugate.apply(qp, q)
        qnsq = qnormsq(q)
        grad_qnsq = (-grad_output * qpq / qnsq**2).sum(-2, keepdim=True) 
        grad_qpq = grad_output / qnsq # correct
        dqpq_dqp = q2m(q, Qmt2)
        grad_pq = dqpq_dqp.matmul(grad_qpq) # correct
        grad_p = q2m(q, Qmt).transpose(-1, -2).matmul(grad_pq)
        
        dqpq_dq = q2m(qp, Qmt3) 
        dpq_dq = q2m(p, Qmt2)
        grad_q = dpq_dq.transpose(-1, -2).matmul(grad_pq) + \
                 dqpq_dq.transpose(-1, -2).matmul(grad_qpq) + \
                 grad_qnsq * 2 * q
        return grad_p, grad_q

def checkGrad():
    torch.manual_seed(42)
    p = torch.rand(1, 4, 1, requires_grad=True).double()
    q = torch.rand(1, 4, 1, requires_grad=True).double()
    from torch.autograd import gradcheck
    global transposer
    transposer = transposer.double()
    test = gradcheck(QConjugate.apply, (p, q), eps=1e-6, atol=1e-4)
    print(test)
    

def qnorm(q): # (*, 4, 1)
    return (q.squeeze(-1)**2).sum(-1).sqrt()

def rotate(q, r):
    '''
    |q|: dims (*, 4, 1)
    |r|: dims (4, 1) 
    '''
    return QMultiply.apply(r, QMultiplyConjugate.apply(q, r))/qnormsq(r)

class QConv1d(nn.Module):
    def __init__(self, inchannels, outchannels, filterlen, stride=1):
        '''
        |inchannels|: number of input quaternion channels
        |outchannels|: number of output quaternion channels
        |filterlen|: length of convolutional filter, recommended odd
        '''
        super(QConv1d, self).__init__()
        self.filterlen = filterlen
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.stride = stride
        self.register_buffer('eye', torch.tensor([[1],[0],[0],[0]]).float())
        
        # sum a(q + b)(qp + c)q(qp + c)^-1
        dims =  (1, outchannels, inchannels, 1, filterlen, 1, 1)
        he = math.sqrt(2 / filterlen / inchannels)
        self.a = nn.Parameter(torch.randn(*dims) * he)
        self.b = nn.Parameter(torch.randn(*dims) / 2)
        self.c = nn.Parameter(torch.randn(*dims) * math.sqrt(1.3780**2 - 1/4))
        # self.beta = nn.Parameter(torch.randn(1, outchannels, 1, 1, 1))

    def forward(self, x):
        '''
        |x|: dims (batch, in channels, in time, 4, 1)
        returns: dims (batch, out channels, out time, 4, 1)
        '''
        #import pdb; pdb.set_trace()
        q = x.unsqueeze(1)
        # q: (batch, 1, in channels, in time, 4, 1)
        
        qp = q.transpose(-3, 0)[self.filterlen//2:-(self.filterlen//2):self.stride].transpose(0, -3).unsqueeze(-3)
        # qp: (batch, 1, in channels, out time, 1, 4, 1)
        
        qpc = qp + self.c*self.eye # (4, 1)                                
        # self.c*self.eye: (1, outchannels, inchannels, 1, filterlen, 4, 1)
        # qpc: (batch, outchannels, inchannels, out time, filterlen, 4, 1)
        
        q = q.unfold(-3, self.filterlen, self.stride).transpose(-3, -1).transpose(-1,-2) 
        # q: (batch, 1, inchannels, out time, filterlen, 4, 1)
        
        res = QMultiply.apply(qpc, QMultiplyConjugate.apply(q, qpc))/qnormsq(qpc)
        #res = QConjugate.apply(q, qpc)
        res = QMultiply.apply(q, res) + self.b * res
        res = res * self.a
        res = res.sum(-3).sum(2)
        # res: dims (batch, out channels, out time, 4, 1)
        return res

class QBatchNorm1d(nn.Module):
    def __init__(self, *dims, momentum=0.1):
        super(QBatchNorm1d, self).__init__()
        
        self.register_buffer('mean', torch.ones(1, *dims, 1, 1, 1))
        self.momentum = momentum

    def forward(self, x):
        '''
        |x|: dims (batch, *dims, time, 4, 1)
        returns: dims (batch, *dims, time, 4, 1)
        '''
        if self.training:
            self.mean = self.mean * (1-self.momentum) + self.momentum * qnormsq(x.detach()).mean(dim=0, keepdim=True).mean(dim=-3, keepdim=True).sqrt()

        return x / self.mean
    
def checkEquivariant():
    x = torch.randn(1,1,7,4,1)
    kernel = QConv1d(1, 1, 7)
    r = torch.randn(4,1)
    x_rot = rotate(x, r)
    fx_rot = kernel(x_rot)
    fx = kernel(x)
    rot_fx = rotate(fx, r)
    frac = (fx_rot*rot_fx).sum()/fx_rot.norm()/rot_fx.norm()
    print(frac)

#checkEquivariant()
#checkGrad()