"""Five band-shared cubic B-spline activations, not edge-independent KAN."""
import torch
from torch import nn
from torch.nn import functional as F

class GroupBSpline(nn.Module):
    def __init__(self, groups=5, grid_size=5, degree=3):
        super().__init__()
        self.groups=groups; self.grid_size=grid_size; self.degree=degree
        grid=torch.arange(-degree,grid_size+degree+1,dtype=torch.float32)*(2/grid_size)-1
        self.register_buffer('grid',grid)
        self.coefficients=nn.Parameter(torch.empty(groups,grid_size+degree))
        self.base_scale=nn.Parameter(torch.ones(groups))
        self.spline_scale=nn.Parameter(torch.ones(groups))
        # Small random spline residual around SiLU, fitted at the interior knots.
        with torch.no_grad():
            knots=grid[degree:-degree]
            basis=self.basis(knots)
            noise=(torch.rand(grid_size+1,groups)-.5)*(.1/grid_size)
            self.coefficients.copy_(torch.linalg.lstsq(basis.double(),noise.double(),driver='gelsd').solution.T.float())

    def basis(self,x):
        z=x.unsqueeze(-1);t=self.grid
        b=((z>=t[:-1]) & (z<t[1:])).to(x.dtype)
        for k in range(1,self.degree+1):
            b=(z-t[:-(k+1)])/(t[k:-1]-t[:-(k+1)])*b[...,:-1]+(t[k+1:]-z)/(t[k+1:]-t[1:-k])*b[...,1:]
        return b

    def forward(self,x):
        if x.shape[-1]!=self.groups: raise ValueError('One activation per frequency band required')
        spline=(self.basis(x)*self.coefficients).sum(-1)
        return self.base_scale*F.silu(x)+self.spline_scale*spline


class EdgeRational(nn.Module):
    """Independent (5,4) rational coefficients for every input-output edge."""
    def __init__(self, source, inputs, outputs):
        super().__init__()
        self.weight_numerator=nn.Parameter(source.weight_numerator.detach().reshape(1,1,6).expand(inputs,outputs,6).clone())
        self.weight_denominator=nn.Parameter(source.weight_denominator.detach().reshape(inputs,1,4).expand(inputs,outputs,4).clone())

    def forward(self,x):
        z=x.unsqueeze(-1)
        a,b=self.weight_numerator,self.weight_denominator.abs()
        numerator=a[...,5]
        for k in range(4,-1,-1):numerator=numerator*z+a[...,k]
        az=z.abs();denominator=b[...,3]
        for k in range(2,-1,-1):denominator=denominator*az+b[...,k]
        denominator=denominator*az+1
        return numerator/denominator

class EdgeRationalLinear(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.lin=source.lin
        self.kan=EdgeRational(source.kan,self.lin.in_features,self.lin.out_features)

    def forward(self,x):
        return (self.kan(x)*self.lin.weight.T).sum(-2)+self.lin.bias
