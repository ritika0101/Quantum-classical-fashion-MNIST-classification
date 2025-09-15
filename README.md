# Fashion-MNIST Hybrid (CNN → PCA → VQC) Classifier

This project trains a **CNN** on Fashion-MNIST, uses its **penultimate dense layer** as a **feature extractor**, compresses features with **PCA** to `n_qubits`, and feeds them to a **variational quantum classifier (VQC)** built with **PennyLane** and wrapped as a **KerasLayer**. A small **classical baseline** on the same PCA features is also included.

> Default dataset in the notebook is **`tf.keras.datasets.fashion_mnist`** 

---

## Pipeline

1. **Data load & split**
   - Fashion-MNIST → `(x_train, y_train), (x_test, y_test)`  
   - Normalize pixels to `[0, 1]`.
   - Train/validation split on the training set (10% validation).

2. **CNN feature extractor (Keras/TensorFlow)**
   - Conv(16, 3×3, ReLU, L2=1e-3) → MaxPool →  
     Conv(32, 3×3, ReLU, L2=1e-3) → MaxPool →  
     Flatten → Dropout(0.3) → Dense(64, ReLU) → Dense(10, Softmax)
   - Train with `adam`, `categorical_crossentropy`, `EarlyStopping(patience=3)`.
   - Build a **feature extractor** model that outputs the **Dense(64)** layer (index -2).

3. **Feature scaling & PCA (scikit-learn)**
   - `StandardScaler` on extracted features (train fit → transform val/test).
   - `PCA(n_components = n_qubits)` to compress to the number of quantum wires.

4. **Angle scaling for quantum encoding**
   - Scale PCA features to **[−π, π]** so they can be fed to **RY** rotations

5. **Variational Quantum Classifier (PennyLane)**
   - Device: `default.qubit`, `wires = n_qubits`
   - **Encoding:** per-qubit `RY(inputs[i])` (angle embedding).
   - **Variational block:** `qml.templates.StronglyEntanglingLayers(weights, wires=...)`
   - **Readout:** expectations `[⟨Z_0⟩, …, ⟨Z_{n_qubits-1}⟩]` (dimension = `n_qubits`).

6. **Hybrid Keras model**
   - `inp: (n_qubits,) → KerasLayer(quantum_circuit) → Dropout(0.3) → Dense(num_classes, softmax)`
   - Optimizer: `Adam(lr=1e-4)`, Loss: `categorical_crossentropy`, Metric: `CategoricalAccuracy`
   - `EarlyStopping(patience=5, restore_best_weights=True)`
   - Train on `(X_train_scaled, y_train_onehot)`, validate on `(X_val_scaled, y_val_onehot)`.

---

## Hyperparameters (where to tweak) ##

**CNN**  
- Conv filters: 16 → 32  
- Dense(64), Dropout(0.3), L2=1e-3   
- batch_size=64, epochs=20, patience=3  

**PCA / Quantum**  
n_qubits (default 4) — trades off dimension vs circuit depth  
n_layers (default 4) — more layers increase expressivity  

**Hybrid training**
`lr=1e-4`, `batch_size=8`, `epochs=20`, `patience=5`  

**Baseline**  
Hidden units: 32, Dropout(0.5), epochs=10, batch_size=32  
