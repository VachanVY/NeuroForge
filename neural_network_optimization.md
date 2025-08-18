# Neural Network Optimization
## Momentum
* <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d4e66c8d-ee41-403a-a7ce-add3a707b2e3" />
* <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9a192232-04cc-4467-8b51-52af23c558eb" />
---
## Bias Correction
* <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1c04b78d-52de-4bfc-a70d-fb0abfbd34a8" />
* <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/fa934ea4-dfb4-4d91-afaf-f42ba7530db2" />
---
* <img width="1913" height="494" alt="image" src="https://github.com/user-attachments/assets/67e03ce6-6096-4396-ab29-81ddf496ca49" />
These up and down oscillations slow down gradient descent, preventing the use of large learning rates.
* <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6ed4bc57-a963-4305-92cb-ebfe251062aa" />
---
## Adaptive Learning Rate
* <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1ca8b59b-7b41-457d-bea9-53f080954a01" />
* <img width="1599" height="1000" alt="image" src="https://github.com/user-attachments/assets/b016c0c5-3020-4c09-b10c-92910bb4cc1a" />
 <details>
  <summary> Difference Between RMSProp and AdaGrad (ChatGpt) </summary>
  <img width="894" height="1024" alt="image" src="https://github.com/user-attachments/assets/1035d364-177d-4d93-a9ac-228d75ed63b4" />
 </details>

---

## Nestrov accelerated gradient
*   <img width="940" height="651" alt="Screenshot from 2025-08-17 18-15-34" src="https://github.com/user-attachments/assets/e3a988f8-2faf-43be-a6f0-4abcce9e4fd3" />
   
*  <img width="701" height="222" alt="image" src="https://github.com/user-attachments/assets/6672e6ea-b438-48e2-b0b1-207ca9dc4b24" />
   
*   <img width="620" height="656" alt="image" src="https://github.com/user-attachments/assets/7bf5dc27-10e2-4038-ab30-d567493e4a5d" />

 
---

## Adam
  <img width="1914" height="998" alt="image" src="https://github.com/user-attachments/assets/8553b369-d9e4-4364-90d6-0f28076e52af" />

---

## Second-order methods
* 

---

## [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon)
* <details>
    <summary> SVD review </summary>
    <img width="1521" height="1032" alt="image" src="https://github.com/user-attachments/assets/d5a57c38-2972-48f4-90af-05e2e4731280" />
  </details>
* <img width="926" height="589" alt="image" src="https://github.com/user-attachments/assets/9533862a-b624-48a8-90d7-53174cef513e" />
* Instead of applying the "momentumized" gradient directly, Muon modifies it using an orthogonalization step and then uses it for weight update
* The orthogonalization process replaces the update matrix G with its nearest orthogonal matrix, which mathematically is equivalent to replacing G with UV^T where USV^T is its SVD. This means:
  You keep the same subspace spanned by the update (the U and V matrices)
  But you remove the singular value scaling (the S matrix) that was causing some directions to dominate
* The intuition is that neural networks benefit from exploring the parameter space more uniformly across different directions rather than being heavily biased toward just the strongest gradient directions. By orthogonalizing, you ensure that:
   * Important but subtle learning directions don't get overwhelmed by dominant ones
   * The optimization process explores the parameter space more efficiently
   * You avoid getting stuck in narrow valleys where only a few directions matter
* Computing SVD is expensive, so they use the Newton–Schulz iteration algorithm
  <img width="684" height="708" alt="image" src="https://github.com/user-attachments/assets/a2273a7e-1071-4777-92fa-cc7899bfcf58" />
  <img width="710" height="698" alt="image" src="https://github.com/user-attachments/assets/178c030e-79eb-4315-afaa-b7c79b708edd" />
  <img width="479" height="693" alt="image" src="https://github.com/user-attachments/assets/5880fecd-005f-4401-af7d-f154ea2bd3c2" />


  <details>
  <summary> Intuition-- talking to ChatGPT </summary>
   <img width="354" height="694" alt="image" src="https://github.com/user-attachments/assets/9dede82c-184b-4c5a-b7ff-64c079d58b3c" />
  
   <img width="489" height="789" alt="image" src="https://github.com/user-attachments/assets/b4fb567c-02fc-49f4-aa4a-7dde2786a417" />
   <img width="509" height="770" alt="image" src="https://github.com/user-attachments/assets/a25e4a42-c0ca-4a08-9036-16317840b293" />
  </details>





---
## Additional Notes on Neural Network Optimization
* <img width="871" height="439" alt="image" src="https://github.com/user-attachments/assets/340396ac-7f02-4fa4-ac69-ce1b76fa7f08" />

---

## References
* ["An overview of gradient descent optimization algorithms"](https://arxiv.org/pdf/1609.04747)
* ["https://cs231n.github.io/neural-networks-3"](https://cs231n.github.io/neural-networks-3)
* ["An updated overview of recent gradient descent algorithms"](https://johnchenresearch.github.io/demon)
* ["deriving-muon"](https://jeremybernste.in/writing/deriving-muon)
* ["Muon: An optimizer for hidden layers in neural networks" by Keller Jordan](https://kellerjordan.github.io/posts/muon/)
<img width="825" height="625" alt="image" src="https://github.com/user-attachments/assets/dda2d910-b3ea-43d6-8a93-eea273da3ef8" />
