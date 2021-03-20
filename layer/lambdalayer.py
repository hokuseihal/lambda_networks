import torch
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self,d,k=16,u=1,h=4,r=23,stride=1):
        super(Lambda,self).__init__()
        self.h=h
        self.k=k
        self.u=u
        assert d%h==0
        v=d//h
        self.v=v
        self.to_qkv=nn.Conv2d(d,h*k+k*u+v*u,1,bias=False)
        self.qbn=nn.BatchNorm2d(h*k)
        self.vbn=nn.BatchNorm2d(u*v)
        assert r%2==1
        self.embedings=nn.Conv3d(u,k,(1,r,r),padding=(0,(r-1)//2,(r-1)//2))
        self.stride=stride

    def forward(self,x):
        B,C,H,W=x.shape
        q,k,v=self.to_qkv(x).split([self.h*self.k,self.k*self.u,self.v*self.u],dim=1)
        q=self.qbn(q).reshape(B,self.h,self.k,-1)
        v=self.vbn(v).reshape(B,self.u,self.v,-1)
        k=torch.softmax(k.reshape(B,self.u,self.k,-1),dim=-1)

        c_lam=torch.einsum('bukn, buvn -> bkv',k,v)
        p_lam=self.embedings(v.view(B,self.u,self.v,H,W)).view(B,self.k,self.v,-1)
        c_y=torch.einsum('bhkn,bkv->bhvn',q,c_lam)
        p_y=torch.einsum('bhkn,bkvn->bhvn',q,p_lam)
        ret=(c_y+p_y).reshape(B,C,H,W)
        if self.stride!=1:
            ret=F.interpolate(ret,scale_factor=1/self.stride)
        return ret

if __name__=='__main__':
    model=Lambda(128)
    out=torch.randn(3,128,64,64)
    out=model(out)
    print(out.shape)