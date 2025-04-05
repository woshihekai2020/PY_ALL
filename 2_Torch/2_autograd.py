##https://www.pytorch123.com/SecondSection/autograd_automatic_differentiation/
# PyTorch 自动微分
import torch

########################################################################################### 1: tensor follow compute EXP
print("\n 1: 创建一个张量，设置requires_grad=True来跟踪与它相关的计算")
x = torch.ones( 2, 2, requires_grad= True )
print( " ~~torch.ones(2, 2, requires_grad= True): \n  ", x )

y = x + 2
print( " \n 针对张量做一个操作")
print( " ~~y = x + 2, y = : \n  ", y )
print( " y作为操作结果被创建，所以它有grad_fn")
print( " ~~y.grad_fn = : \n  ", y.grad_fn )

z = y * y * 3
print("\n ~~z = y * y * 3, z = \n  ", z )
print( " ~~z.grad_fn = : \n  ", z.grad_fn )

out = z.mean()
print( "\n ~~out = z.mean(), out = : \n  ", out )
print( " ~~out.grad_fn = : \n  ", out.grad_fn )

print( "\n .requires_grad_( ... ) 会改变张量的 requires_grad 标记。" )
print( " 如果没有提供相应的参数。输入的标记默认为 False。\n" )

######################################################################################### 2: check grad label every step
print("\n\n\n\n 2: check grad set label every step")
a = torch.randn(2, 2)
a = ( (a * 3) / (a - 1) )
print( "~~a.requires_grad(True) = \n  ", a.requires_grad )

#here, change the content of a
a.requires_grad_( True )
print( "~~改变内容:a.requires_grad_(True) = : \n  ",a.requires_grad )

b = (a * a).sum()
print( "~~b = (a * a).sum(), b.grad_fn = : \n  ", b.grad_fn )


print( "~~out = (y * y * 3).mean().backward() = : \n  ", out.backward() )
print( "~~d(out)/dx = x.grad = : \n  ", x.grad )

########################################################################################### 3: Jacobi vector product EXP
print( "\n\n\n\n Jacobi vector product EXP: " )
x = torch.randn( 3, requires_grad= True )
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print( "\n ~~iter(y = y * 2) = :\n  ", y )

v = torch.tensor( [0.1, 1.0, 0.001], dtype= torch.float )
y.backward( v )
print( "\n ~~x.grad = :\n  ", x.grad )

print( "\n ~~x.requires_grad: \n  ", x.requires_grad )
print( "\n ~~(x ** 2).requires_grad: \n  ", (x ** 2).requires_grad )

with torch.no_grad():
    print( "\n ~~use with torch.no_grad() to stop auto grad")
    print( " ~~(x ** 2).requires_grad : \n  ", (x ** 2).requires_grad )