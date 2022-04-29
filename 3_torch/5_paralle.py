
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size  = 5
output_size = 2

batch_size = 30
data_size  = 100

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )

class RandomDataset( Dataset ):
    def __init__( self, size, length ):
        self.len  = length
        self.data = torch.randn( length, size )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader( dataset= RandomDataset(input_size, data_size), batch_size= batch_size, shuffle= True )
#获得一个输入，执行一个线性操作，然后给一个输出。尽管如此，你可以使用 DataParallel   在任何模型(CNN, RNN, Capsule Net 等等.)

#创建模型并且数据并行处理
class Model( nn.Module ):
    def __init__(self, input_size, output_size ):
        super( Model, self ).__init__()
        self.fc = nn.Linear( input_size, output_size )

    def forward(self, input):
        output = self.fc( input )
        print( "\t In Model: input size", input.size(), "output size", output.size() )
        return output

model = Model( input_size, output_size )
print( "device_count: ", torch.cuda.device_count() )
if torch.cuda.device_count() > 1 :
    print( "let's use", torch.cuda.device_count(), "GPUs!" )
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel( model )
model.to( device )


#现在我们可以看到输入和输出张量的大小了。
for data in rand_loader:
    input  = data.to(device)
    output = model( input )
    print( "outside: input size", input.size(), "output size", output.size() )

























