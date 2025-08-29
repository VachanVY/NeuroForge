* <img width="878" height="377" alt="image" src="https://github.com/user-attachments/assets/f3706ba1-24fd-458a-9689-9fe957c444cb" />

# Low Precision Training
* <img width="1029" height="581" alt="image" src="https://github.com/user-attachments/assets/be67cac7-b94c-4c6c-8724-b4065b817f3c" />
* <img width="712" height="476" alt="image" src="https://github.com/user-attachments/assets/e628b3fc-a6ea-4d90-a4ed-1ffca6c1136e" />
* <img width="999" height="746" alt="image" src="https://github.com/user-attachments/assets/480f808e-d6ff-449c-b0da-43e32ab11eb2" />
* <img width="972" height="626" alt="image" src="https://github.com/user-attachments/assets/61ce78e7-acf9-4226-90fc-2c72f3c8c51b" />
* <img width="1247" height="359" alt="image" src="https://github.com/user-attachments/assets/57c0d071-5ae0-4db7-afd7-d7757e3e8859" />
* <img width="817" height="869" alt="image" src="https://github.com/user-attachments/assets/a6f52ab3-2acb-457e-932a-ef4a980ffcfe" />
* <img width="1619" height="679" alt="image" src="https://github.com/user-attachments/assets/075578c9-a855-481a-9a81-59c06e9ee07b" />
* <img width="1718" height="681" alt="image" src="https://github.com/user-attachments/assets/3fcefae6-a6fd-4a28-8cf7-6c18c6fd8dbe" />
* [Automatic Mixed Precision -- PyTorch Docs](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

## FP8 Formats for Deep Learning
* <img width="778" height="196" alt="image" src="https://github.com/user-attachments/assets/32ccf84f-5ac6-4829-804d-541ab3c11e09" />
* Some networks survive with one global scale, but others (esp. Transformers w/ residuals) don’t.
  Example: GPT-3 inference with FP8 residuals: global bias gave perplexity 12.59 vs baseline 10.19.
  With per-tensor scaling => 10.29 (basically baseline).
  Per-tensor scaling is mandatory for robustness
* FP8 is too risky for accumulation → keep FP16/BF16 for math.
  However, FP8 is ideal for storage and interchange where tensors are frequently moved around.
  By carefully scaling each tensor, they squeeze FP16 accuracy into FP8 representation.
  This makes training faster (reducing memory bandwidth) and cheaper (reducing storage requirements) without affecting learning dynamics.
* Didn’t compute in FP8, only stored inputs/outputs in FP8. All maths was FP16/BF16/FP32

### [Using FP8 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html?utm_source=chatgpt.com)
* <img width="916" height="991" alt="image" src="https://github.com/user-attachments/assets/f581bb64-721a-4f13-bcd1-cd7e783b2a1e" />
