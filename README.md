# Network_Compression_SVD
Neural Network Compression using Singular Value Decomposition (SVD)
# Neural Network Compression using Singular Value Decomposition (SVD)

## 🚀 Overview  
Deep learning models, especially fully connected networks, often have **redundant parameters**. This project investigates **model compression** using **Singular Value Decomposition (SVD)** to reduce **memory and computational costs** while maintaining accuracy.

### **📝 Project Goals:**  
- Train a **Fully Connected Neural Network (FCNN)** on **MNIST** as a baseline model.  
- Use **SVD** to decompose and **approximate weight matrices** with fewer parameters.  
- Experiment with **D-values** (number of singular values retained) to analyze compression trade-offs.  
- Fine-tune compressed models to recover accuracy.  
---

##  What We Tried to Prove:
- Lower-rank approximations of weight matrices can significantly reduce model size while maintaining high accuracy.
- D-values control the trade-off between model compression and classification accuracy.
- Fine-tuning the compressed models helps recover lost accuracy, making them nearly as effective as the uncompressed baseline model.

---

## 📂 Dataset  
**MNIST:** Handwritten digit dataset with **60,000 training** and **10,000 test** images (size: **28×28 pixels**).  

---

## 🏗️ Implemented Steps  

### **1️⃣ Baseline Model Training**  
✅ A **deep Fully Connected Neural Network (FCNN)** trained on **MNIST**.  
✅ Layers: **5 hidden layers** with **1024 neurons each** + Softmax output.  
✅ **Adam optimizer + ReLU activation + Cross-Entropy Loss.**  

### **2️⃣ SVD-based Compression**  
✅ Apply **SVD** on weight matrices of hidden layers.  
✅ **Retain only the top D singular values** to approximate the original weight matrix.  
✅ **Test different D-values** to balance **compression vs. accuracy**.  

---

## 📊 **Impact of Different D-values**  

| **D-Value** | **Compression Level** | **Test Accuracy (%)** |
|------------|--------------------|--------------------|
| Full (1024) | No compression (Baseline) | **97.0%** |
| **200** | High accuracy, slight compression | **96.8%** |
| **100** | Balanced accuracy & compression | **95.5%** |
| **50** | Moderate compression | **94.1%** |
| **20** | Strong compression, noticeable accuracy drop | **91.1%** |
| **10** | Very strong compression, accuracy significantly reduced | **87.5%** |

### **Key Observations:**  
✅ **D=200** achieves near-baseline accuracy (**96.8%**) with noticeable compression.  
✅ **D=50** provides **moderate compression** with **only a small accuracy drop** (~94.1%).  
✅ **D=10 & D=20** show significant compression but **accuracy drops more**.  
✅ **Fine-tuning improves performance**, helping compressed models **recover lost accuracy**.  

---

## ⚙️ Dependencies  
Ensure you have the following installed before running the project:  

```bash
pip install tensorflow numpy matplotlib
