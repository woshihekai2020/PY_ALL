import torch
#如果你想计算导数，你可以调用 Tensor.backward()。
# 如果 Tensor 是标量（即它包含一个元素数据），则不需要指定任何参数backward()，
# 但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。

#将其属性 .requires_grad 设置为 True，则会开始跟踪针对 tensor 的所有操作。
x = torch.ones( 2, 2, requires_grad= True )
print( "\n x = :\n", x )
#y 作为操作的结果被创建，所以它有 grad_fn
y = x + 2
print( "\n y = :\n" ,y )
print( "\n y.grad_fn = :\n", y.grad_fn )

z = y * y * 3
out = z.mean()
print( "\n z, out = :\n", z, out )

#.requires_grad_( ... ) 会改变张量的 requires_grad 标记。
# 输入的标记默认为 False ，如果没有提供相应的参数。
a = torch.randn( 2, 2 )
a = ((a * 3) / (a - 1))
print( "\n a.requires_grad = : \n", a.requires_grad )
a.requires_grad_(True)
print( "\n a.requires_grad = : \n", a.requires_grad )
b = (a * a).sum()
print( "\n b.grad_fn = : \n",b.grad_fn )


#完成计算后，您可以调用 .backward() 来自动计算所有梯度。
# 该张量的梯度将累积到 .grad 属性中。
out.backward()
print( "\n x.grad: \n" , x.grad )


#一个雅可比向量积的例子：
x = torch.randn( 3, requires_grad= True )
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print("\n jacobian vector: \n", y )

#在这种情况下，y 不再是一个标量。torch.autograd 不能够直接计算整个雅可比，
# 但是如果我们只想要雅可比向量积，只需要简单的传递向量给 backward 作为参数。
v = torch.tensor([0.1, 1.0, 0.0001], dtype= torch.float )
y.backward( v )
print( "\n y.backward(v): \n", x.grad )


#可以通过将代码包裹在 with torch.no_grad()，
#来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导
print( "\n", x.requires_grad )
print( '\n', (x **2).requires_grad )
with torch.no_grad():
    print( (x ** 2 ).requires_grad )