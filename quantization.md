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
* But the lack of offset restricts the mapping between integer and floating-point domain. As a result, the choice of signed or unsigned integer grid matters:
  <img width="891" height="628" alt="image" src="https://github.com/user-attachments/assets/01c51e01-45e9-4c0b-9000-7c76d12b792f" />

