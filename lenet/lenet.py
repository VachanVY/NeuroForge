import torch
from torch import Tensor

from nn import Module, Conv2d, Maxpool2d, Linear, ReLU, Reshape, Softmax, CrossEntropy


class LeNet(Module):
    """
    LeNet-5 implementation using custom neural network API.
    
    Architecture:
    - Conv2d: 1 -> 6 channels, 5x5 kernel, stride 1
    - MaxPool2d: 2x2, stride 2
    - Conv2d: 6 -> 16 channels, 5x5 kernel, stride 1
    - MaxPool2d: 2x2, stride 2
    - Flatten
    - Linear: 16*5*5 -> 120
    - ReLU
    - Linear: 120 -> 84
    - ReLU
    - Linear: 84 -> 10
    - Softmax (for classification)
    """
    
    def __init__(self, num_classes:int = 10):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), strides=(1, 1), bias=True)
        self.pool1 = Maxpool2d(kernel_size=(2, 2), strides=(2, 2))
        
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), strides=(1, 1), bias=True)
        self.pool2 = Maxpool2d(kernel_size=(2, 2), strides=(2, 2))
        
        # Flatten layer - for 32x32 input: (32-5+1)/1 = 28, 28/2 = 14, (14-5+1)/1 = 10, 10/2 = 5
        # So we have 16 * 5 * 5 = 400 features
        # But we need to handle variable batch size, so use -1 for batch dimension
        self.flatten = Reshape(shape=(-1, 400))
        
        # Classification layers
        self.fc1 = Linear(fan_in=400, fan_out=120)
        self.relu1 = ReLU()
        
        self.fc2 = Linear(fan_in=120, fan_out=84)
        self.relu2 = ReLU()
        
        self.fc3 = Linear(fan_in=84, fan_out=num_classes)
        self.softmax = Softmax()
        
        # Loss function
        self.criterion = CrossEntropy()
        
        # Store intermediate values for backward pass
        self.activations = {}
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through LeNet.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 32)
            
        Returns:
            Output probabilities of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.conv1(x)
        self.activations['conv1'] = x
        
        x = self.pool1(x)
        self.activations['pool1'] = x
        
        x = self.conv2(x)
        self.activations['conv2'] = x
        
        x = self.pool2(x)
        self.activations['pool2'] = x
        
        # Flatten
        x = self.flatten(x)
        self.activations['flatten'] = x
        
        # Classification
        x = self.fc1(x)
        self.activations['fc1'] = x
        
        x = self.relu1(x)
        self.activations['relu1'] = x
        
        x = self.fc2(x)
        self.activations['fc2'] = x
        
        x = self.relu2(x)
        self.activations['relu2'] = x
        
        x = self.fc3(x)
        self.activations['fc3'] = x
        
        x = self.softmax(x)
        self.activations['softmax'] = x
        
        return x
    
    def backward(self, y_true: Tensor) -> dict:
        """
        Backward pass through LeNet.
        
        Args:
            y_true: True labels of shape (batch_size,)
            
        Returns:
            Dictionary containing gradients for all parameters
        """
        # Start with loss gradient
        dL_dO = self.criterion.backward()
        
        # Backward through softmax
        dL_dO = self.softmax.backward(dL_dO)
        
        # Backward through fc3
        dL_dO, dL_dw3, dL_db3 = self.fc3.backward(self.activations['relu2'], dL_dO)
        
        # Backward through relu2
        dL_dO = self.relu2.backward(dL_dO)
        
        # Backward through fc2
        dL_dO, dL_dw2, dL_db2 = self.fc2.backward(self.activations['relu1'], dL_dO)
        
        # Backward through relu1
        dL_dO = self.relu1.backward(dL_dO)
        
        # Backward through fc1
        dL_dO, dL_dw1, dL_db1 = self.fc1.backward(self.activations['flatten'], dL_dO)
        
        # Backward through flatten
        dL_dO = self.flatten.backward(dL_dO)
        
        # Backward through pool2
        dL_dO = self.pool2.backward(dL_dO)
        
        # Backward through conv2
        dL_dO, dL_dw_conv2, dL_db_conv2 = self.conv2.backward(dL_dO)
        
        # Backward through pool1
        dL_dO = self.pool1.backward(dL_dO)
        
        # Backward through conv1
        dL_dinput, dL_dw_conv1, dL_db_conv1 = self.conv1.backward(dL_dO)
        
        # Return gradients
        return {
            'input_grad': dL_dinput,
            'conv1_weight_grad': dL_dw_conv1,
            'conv1_bias_grad': dL_db_conv1,
            'conv2_weight_grad': dL_dw_conv2,
            'conv2_bias_grad': dL_db_conv2,
            'fc1_weight_grad': dL_dw1,
            'fc1_bias_grad': dL_db1,
            'fc2_weight_grad': dL_dw2,
            'fc2_bias_grad': dL_db2,
            'fc3_weight_grad': dL_dw3,
            'fc3_bias_grad': dL_db3,
        }
    
    def compute_loss(self, x: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute forward pass and loss.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 32)
            y_true: True labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        y_pred = self.forward(x)
        loss = self.criterion.forward(y_true, y_pred)
        return loss


def test_lenet():
    """Test LeNet with random data and compare gradients with PyTorch"""
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = LeNet(num_classes=10).to(device)
    
    # Create sample data (MNIST-like: 32x32 grayscale images)
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 32).to(device)
    y_true = torch.randint(0, 10, (batch_size,)).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels: {y_true}")
    
    # Forward pass
    y_pred = model.forward(x)
    print(f"Output shape: {y_pred.shape}")
    
    # Compute loss
    loss = model.compute_loss(x, y_true)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    gradients = model.backward(y_true)
    print("Backward pass completed successfully!")

    # Compare with PyTorch
    print("\n--- Comparing with PyTorch ---")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchLeNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    torch_model = TorchLeNet(num_classes=10)

    # Copy weights
    torch_model.conv1.weight.data = model.conv1.wei.clone()
    if model.conv1.bias is not None and torch_model.conv1.bias is not None:
        torch_model.conv1.bias.data = model.conv1.bias.clone()
    torch_model.conv2.weight.data = model.conv2.wei.clone()
    if model.conv2.bias is not None and torch_model.conv2.bias is not None:
        torch_model.conv2.bias.data = model.conv2.bias.clone()
    torch_model.fc1.weight.data = model.fc1.wie.clone()
    torch_model.fc1.bias.data = model.fc1.bias.clone()
    torch_model.fc2.weight.data = model.fc2.wie.clone()
    torch_model.fc2.bias.data = model.fc2.bias.clone()
    torch_model.fc3.weight.data = model.fc3.wie.clone()
    torch_model.fc3.bias.data = model.fc3.bias.clone()

    # PyTorch forward and backward
    torch_x = x.clone().requires_grad_()
    torch_output = torch_model(torch_x)
    
    # PyTorch uses CrossEntropyLoss which combines LogSoftmax and NLLLoss.
    # To compare, we can use our softmax output and compute NLL loss.
    # Or, more simply, use the logits from our fc3 and torch's fc3 with CrossEntropyLoss.
    
    # Let's get our logits (pre-softmax)
    our_logits = model.activations['fc3']
    
    torch_loss_fn = nn.CrossEntropyLoss()
    torch_loss = torch_loss_fn(torch_output, y_true)
    torch_loss.backward()

    print(f"PyTorch Loss: {torch_loss.item():.4f}")

    # Compare gradients
    print("\n--- Gradient Comparison ---")
    
    def compare_grads(name, our_grad, torch_grad):
        if our_grad is None and torch_grad is None:
            print(f"✅ {name}: Both grads are None.")
            return
        if our_grad is None or torch_grad is None:
            print(f"❌ {name}: Mismatch - one grad is None.")
            return
            
        is_close = torch.allclose(our_grad, torch_grad, atol=1e-5)
        status = "✅" if is_close else "❌"
        print(f"{status} {name}: All close: {is_close}")
        if not is_close:
            print(f"   - Max diff: {torch.max(torch.abs(our_grad - torch_grad))}")
            # print("Our grad:\n", our_grad)
            # print("Torch grad:\n", torch_grad)

    # To get the equivalent of PyTorch's CrossEntropyLoss backward,
    # we need to start our backward pass from the gradient of the loss with respect to the logits.
    # PyTorch's CrossEntropyLoss(logits, y_true) is equivalent to NLLLoss(LogSoftmax(logits), y_true)
    # The gradient dL/d_logits is (softmax(logits) - one_hot(y_true)) / batch_size
    
    y_one_hot = F.one_hot(y_true, num_classes=10).float().to(device)
    dL_dlogits = (y_pred - y_one_hot) / batch_size # y_pred is our softmax output

    # Now, start a custom backward from here
    dL_dO = dL_dlogits
    dL_dO, dL_dw3, dL_db3 = model.fc3.backward(model.activations['relu2'], dL_dO)
    dL_dO = model.relu2.backward(dL_dO)
    dL_dO, dL_dw2, dL_db2 = model.fc2.backward(model.activations['relu1'], dL_dO)
    dL_dO = model.relu1.backward(dL_dO)
    dL_dO, dL_dw1, dL_db1 = model.fc1.backward(model.activations['flatten'], dL_dO)
    dL_dO = model.flatten.backward(dL_dO)
    dL_dO = model.pool2.backward(dL_dO)
    dL_dO, dL_dw_conv2, dL_db_conv2 = model.conv2.backward(dL_dO)
    dL_dO = model.pool1.backward(dL_dO)
    dL_dinput, dL_dw_conv1, dL_db_conv1 = model.conv1.backward(dL_dO)

    custom_grads = {
        'input_grad': dL_dinput,
        'conv1_weight_grad': dL_dw_conv1, 'conv1_bias_grad': dL_db_conv1,
        'conv2_weight_grad': dL_dw_conv2, 'conv2_bias_grad': dL_db_conv2,
        'fc1_weight_grad': dL_dw1, 'fc1_bias_grad': dL_db1,
        'fc2_weight_grad': dL_dw2, 'fc2_bias_grad': dL_db2,
        'fc3_weight_grad': dL_dw3, 'fc3_bias_grad': dL_db3,
    }

    compare_grads("Input", custom_grads['input_grad'], torch_x.grad)
    compare_grads("Conv1 Weight", custom_grads['conv1_weight_grad'], torch_model.conv1.weight.grad)
    if model.conv1.bias is not None and torch_model.conv1.bias is not None:
        compare_grads("Conv1 Bias", custom_grads['conv1_bias_grad'], torch_model.conv1.bias.grad)
    compare_grads("Conv2 Weight", custom_grads['conv2_weight_grad'], torch_model.conv2.weight.grad)
    if model.conv2.bias is not None and torch_model.conv2.bias is not None:
        compare_grads("Conv2 Bias", custom_grads['conv2_bias_grad'], torch_model.conv2.bias.grad)
    compare_grads("FC1 Weight", custom_grads['fc1_weight_grad'], torch_model.fc1.weight.grad)
    compare_grads("FC1 Bias", custom_grads['fc1_bias_grad'], torch_model.fc1.bias.grad)
    compare_grads("FC2 Weight", custom_grads['fc2_weight_grad'], torch_model.fc2.weight.grad)
    compare_grads("FC2 Bias", custom_grads['fc2_bias_grad'], torch_model.fc2.bias.grad)
    compare_grads("FC3 Weight", custom_grads['fc3_weight_grad'], torch_model.fc3.weight.grad)
    compare_grads("FC3 Bias", custom_grads['fc3_bias_grad'], torch_model.fc3.bias.grad)

    return model, x, y_true, y_pred, loss, gradients


if __name__ == "__main__":
    print("Testing LeNet implementation...")
    model, x, y_true, y_pred, loss, gradients = test_lenet()
    print("\nLeNet test completed successfully!")
