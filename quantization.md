# Dtypes used in Neural Network Training
* <img width="1029" height="581" alt="image" src="https://github.com/user-attachments/assets/798d9e82-5d4d-42f5-b277-30d237fc91c8" />

# Neural Network Quantization
* <img width="778" height="554" alt="image" src="https://github.com/user-attachments/assets/4569da37-dd30-4392-bfef-eba1fd3239d8" />
* When moving from 32 to 8 bits, the memory overhead of storing tensors decreases by a factor of 4, while the computational cost for matrix multiplication reduces quadratically by a factor of 16
* Low bit-width quantization introduces noise to the network that can lead to a drop in accuracy. While some networks are robust to this noise, other networks require extra work to exploit the benefits of quantization
---
* <img width="1034" height="747" alt="image" src="https://github.com/user-attachments/assets/02e96803-470b-4c5d-abbd-5d7df9733836" />
* Reducing the scale factor `s` reduces the step size and improves precision, but also shrinks the representable range, making clipping more likely\
  Increasing `s` reduces clipping but increases rounding error
* <img width="991" height="419" alt="image" src="https://github.com/user-attachments/assets/8cc97b2a-a057-4e36-bf9f-f420270e24fd" />
* <img width="1025" height="920" alt="image" src="https://github.com/user-attachments/assets/5561bed7-9741-4d79-8520-85483ec70db5" />
* In neural networks, zero has a special meaning
  <img width="987" height="441" alt="image" src="https://github.com/user-attachments/assets/29476872-2612-47b2-a23b-034920c85efe" />
  <img width="998" height="521" alt="image" src="https://github.com/user-attachments/assets/0949ceac-2012-406c-9ae2-35e3e7b698e5" />
* 
---
## Uniform affine quantization (asymmetric quantization)
* Quantization: `The scale factor s, the zero-point z and the bit-width b`
* <img width="1025" height="735" alt="image" src="https://github.com/user-attachments/assets/a0e81e52-4e28-42c7-a816-3a0c250fa531" />

## Symmetric uniform quantization
* Symmetric quantization is a simplified version of the general asymmetric case. The symmetric quantizer restricts the zero-point to 0\
  This reduces the computational overhead of dealing with zero-point offset during the accumulation operation in equation (3)
* But the lack of offset restricts the mapping between the integer and the floating-point domains. As a result, the choice of signed or unsigned integer grid matters:
  <img width="891" height="628" alt="image" src="https://github.com/user-attachments/assets/01c51e01-45e9-4c0b-9000-7c76d12b792f" />

## Quantization Simulation
* <img width="892" height="561" alt="image" src="https://github.com/user-attachments/assets/fe8154cc-6fde-4913-ac3a-61aa89a57dc8" />
* We aim to approximate fixed-point operations using floating-point hardware. Such simulations are significantly easier to implement compared to running experiments on actual quantized hardware or using quantized kernels
* During on-device inference, all the inputs (biases, weights, and input activations) to the hardware are in a fixed-point format. However, when we simulate quantization using common deep learning frameworks and general-purpose hardware these quantities are in floating-point. This is why we introduce quantizer blocks in the compute graph to induce quantization effects.
* Figure 4b shows how the same convolutional layer is modelled in a deep-learning framework. Quantizer blocks are added in between the weights and the convolution to simulate weight quantization, and after the activation function to simulate activation quantization. The bias is often not quantized because it is stored in higher precision. In section 2.3.2, we discuss in more detail when it is appropriate to position the quantizer after the non-linearity

## Batch normalization folding
* <img width="732" height="1201" alt="image" src="https://github.com/user-attachments/assets/7aff4e6e-0b84-463d-9faf-f792b32acdd0" />

## Activation function fusing
* <img width="881" height="381" alt="image" src="https://github.com/user-attachments/assets/e41c0e71-162b-4b6d-9822-a548a944d932" />

## 
