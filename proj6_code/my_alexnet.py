import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
        to understand what it means

        Note: Do not forget to freeze the layers of alexnet except the last one

        Download pretrained alexnet using pytorch's API (Hint: see the import
        statements)
        '''
        super().__init__()

        self.cnn_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

        model=alexnet(pretrained=True)        
        alex_children = model.children()
        print("alex model summary")
        for l in model.children():
            print(l)
        
        cnn_layers, avg_layer, linear_layers = alex_children
        
        # cnn layers
        self.cnn_layers = nn.Sequential(
                *list(cnn_layers.children()), 
            )
        for param in self.cnn_layers.parameters():
            param.requires_grad = False
        print("cnn model summary")
        print(self.cnn_layers)
        
        # linear layers
        self.fc_layers = nn.Sequential(
                nn.Flatten(),
                *list(linear_layers.children())[:-1],
                nn.Linear(in_features=4096, out_features=15, bias=True)
            )
        self.fc_layers[2].weight.requires_grad = False
        self.fc_layers[2].bias.requires_grad = False
        self.fc_layers[5].weight.requires_grad = False
        self.fc_layers[5].bias.requires_grad = False
#         self.fc_layers[7].weight.requires_grad = False
#         self.fc_layers[7].bias.requires_grad = False
        
        print("fc model summary")
        print(self.fc_layers)
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    
    ###########################################################################
    # Student code end
    ###########################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        '''

        model_output = None
        x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################

        model_output = self.cnn_layers(x)
        model_output = self.fc_layers(model_output)
    
    ###########################################################################
    # Student code end
    ###########################################################################
        return model_output
