## 1. The Sun-Dutta Selection Method

When generating thousands of non-orthogonal determinants (e.g., via finite difference of CISD/CCSD amplitudes), the majority of them offer redundant geometrical and physical information. Including all of them results in an ill-conditioned, computationally explosive Overlap matrix. 

The Sun-Dutta selection method is a greedy, forward-selection algorithm designed to build a compact active space by evaluating candidates one-by-one against the *currently accepted* NOCI wave function ($|\Psi_{\text{active}}\rangle$). 

For each candidate determinant $|\Phi_i\rangle$, the algorithm performs two rigorous checks:

**Step 1: The Geometric Shield (Linear Independence)**
Before calculating any energies, the algorithm checks the orthogonal distance of $|\Phi_i\rangle$ from the active space. It calculates the orthogonal projection remainder (`proj_new`):
$$\text{proj\_new} = 1.0 - \frac{\langle \Phi_i | \mathbf{S}_{\text{inv}} | \Phi_i \rangle}{S_{ii}}$$
If `proj_new` is smaller than a strict tolerance (e.g., $10^{-8}$), the determinant is heavily parallel to the existing active space. It is instantly rejected to prevent the Overlap matrix from becoming singular.

**Step 2: The Physical Evaluation (The $2 \times 2$ Matrix)**
If the determinant passes the geometric shield, a $2 \times 2$ subspace is constructed using the current NOCI state and the new candidate:
$$\mathbf{H}_{2\times2} = \begin{pmatrix} \langle \Psi_{\text{active}} | \hat{H} | \Psi_{\text{active}} \rangle & \langle \Psi_{\text{active}} | \hat{H} | \Phi_i \rangle \\ \langle \Phi_i | \hat{H} | \Psi_{\text{active}} \rangle & \langle \Phi_i | \hat{H} | \Phi_i \rangle \end{pmatrix}$$
This $2 \times 2$ system is diagonalized. If the new lowest eigenvalue $\epsilon_0$ lowers the total energy by more than a predefined threshold ($\Delta E > 10^{-10}$ Ha), the determinant is permanently accepted, and $|\Psi_{\text{active}}\rangle$ is updated.

---

## 2. The Flaw of "Order Bias" and the Need for Sorting

A pure Sun-Dutta algorithm evaluates candidates in the arbitrary order they are generated. This introduces a fatal flaw known as **Order Bias**. 

If the algorithm evaluates a "mediocre" determinant early in the list, it might accept it because it lowers the energy slightly. However, this locks that specific geometry into the active space. When the algorithm later encounters a "fantastic", highly correlated determinant that happens to be geometrically similar to the mediocre one, the Geometric Shield will reject the fantastic determinant! The active space becomes clogged with sub-optimal physics.

**The Solution:**
To cure Order Bias, the candidate pool must be **Sorted** prior to evaluation. 
By ranking the determinants from most physically important to least important, we guarantee that the absolute strongest correlation vectors claim the available geometric space first. 
* **Energy Sort:** Evaluate the $2 \times 2$ energy of every candidate against the HF reference and sort by the energy lowering.
* **Amplitude Proxy:** (Zero-cost) Sort the candidate pool directly by the absolute value of the original CISD/CCSD amplitude that generated the finite-difference vector.
* **Iterative Thresholding (CIPSI-style):** Sweep the unsorted pool multiple times, starting with a massive energy tolerance (e.g., $10^{-3}$) and slowly lowering it to $10^{-10}$. This pseudo-sort ensures the heavy hitters are always accepted in the first sweep.

---

## 3. The Global Geometric Shield (QRCP)

Even with sorting, building a massive NOCI active space using only the local Sun-Dutta geometric check (`proj_new`) is dangerous. Evaluating local orthogonal projections step-by-step slowly accumulates floating-point errors—a phenomenon known as "Gram-Schmidt rot." Eventually, microscopic linear dependencies slip through, the active space Overlap matrix becomes singular, and the variational energy violently collapses.

To physically prevent this, we apply a global mathematical shield *before* running the Sun-Dutta filter using **QR Decomposition with Column Pivoting (QRCP)**.

By running QRCP on the global Overlap matrix ($\mathbf{S}$), the algorithm strictly factors the matrix to find the mathematical rank of the space. It explicitly identifies the minimal set of columns that span the entire geometric space up to a strict tolerance (e.g., $10^{-7}$). Any determinant not in this pivot list is, by mathematical definition, a linear combination of the safe base. By restricting the Sun-Dutta filter to evaluate *only* the vectors pre-approved by the QRCP algorithm, a variational collapse becomes mathematically impossible.

---

## 4. Breaking the Scaling Wall: Matrix-Free Pivoted Cholesky

While global QRCP is mathematically flawless, it hits a hard computational ceiling. QRCP requires building and storing the entire dense $N \times N$ Overlap matrix, which scales as $\mathcal{O}(N^3)$. In multi-reference Resonating Hartree-Fock expansions, the determinant pool can easily exceed 40,000 vectors, making the full matrix evaluation impossible.

To bypass this scaling wall, we replace standard QRCP with **Matrix-Free Pivoted Cholesky Decomposition**.

Because the Overlap matrix $\mathbf{S}$ is a Gram matrix (symmetric and positive semi-definite), applying Pivoted Cholesky decomposition to $\mathbf{S}$ yields the exact same geometric independent subset (the pivot array) as running QRCP on the wavefunctions.

By only calculating the exact physics needed to find the next orthogonal dimension, Matrix-Free Cholesky reduces the scaling of the global geometric shield from $\mathcal{O}(N^3)$ to $\mathcal{O}(N \cdot k^2)$ (where $k$ is the true rank of the space). This allows us to rapidly and safely extract the independent basis from tens of thousands of multireference determinants without memory exhaustion.

---

## 5. Explicit Mathematical Formulations and Proof of Equivalence

To understand why the $\mathcal{O}(N^3)$ QRCP method and the $\mathcal{O}(N)$ Matrix-Free Cholesky method are geometric twins, we must examine the explicit equations that govern them.

### 5.1 QRCP on the Wavefunctions

Consider a theoretical matrix $\mathbf{A}$ of dimensions $M \times N$, where each of the $N$ columns is a generated many-electron Slater determinant. QR Decomposition with Column Pivoting factors $\mathbf{A}$ into an orthogonal matrix $\mathbf{Q}$ and an upper-triangular matrix $\mathbf{R}$, while applying a permutation matrix $\mathbf{P}$ to reorder the columns:
$$\mathbf{A} \mathbf{P} = \mathbf{Q} \mathbf{R}$$

At each step, QRCP selects the column with the maximum $L_2$ norm, swaps it to the front via $\mathbf{P}$, and projects its geometry out of the remaining vectors. 

### 5.2 Pivoted Cholesky on the Gram Matrix

In quantum chemistry, we rarely build the dense wavefunction matrix $\mathbf{A}$. Instead, we compute the Gram matrix (Overlap matrix, $\mathbf{S}$), which is symmetric and positive semi-definite:
$$\mathbf{S} = \mathbf{A}^T \mathbf{A}$$

Pivoted Cholesky Decomposition factors this matrix into a lower-triangular matrix $\mathbf{L}$ and its transpose:
$$\mathbf{P}^T \mathbf{S} \mathbf{P} = \mathbf{L} \mathbf{L}^T$$

### 5.3 Proof of Equivalence

We can prove that these algorithms yield the exact same geometric selection by substituting the QRCP factorization into the definition of the Overlap matrix.
Applying the column permutation $\mathbf{P}$ to both rows and columns of $\mathbf{S}$:
$$\mathbf{P}^T \mathbf{S} \mathbf{P} = \mathbf{P}^T (\mathbf{A}^T \mathbf{A}) \mathbf{P} = (\mathbf{A} \mathbf{P})^T (\mathbf{A} \mathbf{P})$$

Substitute $\mathbf{Q}\mathbf{R}$ in place of $\mathbf{A}\mathbf{P}$:
$$\mathbf{P}^T \mathbf{S} \mathbf{P} = (\mathbf{Q} \mathbf{R})^T (\mathbf{Q} \mathbf{R}) = \mathbf{R}^T \mathbf{Q}^T \mathbf{Q} \mathbf{R}$$

Because $\mathbf{Q}$ is an orthogonal matrix, $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$, causing it to drop out of the equation:
$$\mathbf{P}^T \mathbf{S} \mathbf{P} = \mathbf{R}^T \mathbf{R}$$

Let $\mathbf{L} = \mathbf{R}^T$. Because $\mathbf{R}$ is upper-triangular, $\mathbf{L}$ must be lower-triangular. Substituting this gives the exact definition of Pivoted Cholesky:
$$\mathbf{P}^T \mathbf{S} \mathbf{P} = \mathbf{L} \mathbf{L}^T$$
Therefore, factoring the Overlap matrix $\mathbf{S}$ using Pivoted Cholesky mathematically guarantees the exact same permutation matrix $\mathbf{P}$ as running QRCP on the true wavefunctions.

### 5.4 The Matrix-Free Cholesky Algorithm

The Matrix-Free variant computes $\mathbf{L}$ column by column without ever building the dense global matrix $\mathbf{S}$.

**Step 0: Initialize the Diagonals**
Let $\mathbf{d}$ be an array of length $N$ representing the active diagonal of $\mathbf{S}$ (the initial self-overlap):
$$d_i^{(0)} = S_{ii} \quad \text{for } i = 1 \dots N$$

**Step 1: The Pivot Selection**
Find the determinant with the largest remaining orthogonal distance (Schur complement) to be the pivot $p_k$:
$$p_k = \arg\max_{i} \left( d_i^{(k-1)} \right)$$

**Step 2: Lazy Column Evaluation**
Compute only the single column of overlaps between the chosen pivot $p_k$ and all other determinants $i$:
$$S_{i, p_k} = \langle \Phi_i | \Phi_{p_k} \rangle$$

**Step 3: Compute the Cholesky Vector**
Calculate the $k$-th column of $\mathbf{L}$ by subtracting the geometric overlap accounted for in previous steps:
$$L_{i,k} = \frac{S_{i, p_k} - \sum_{j=0}^{k-1} L_{i,j} L_{p_k, j}}{\sqrt{d_{p_k}^{(k-1)}}}$$

**Step 4: Update the Diagonals (The Schur Complement)**
Update the remaining diagonals by projecting out the geometric volume captured by the new vector:
$$d_i^{(k)} = d_i^{(k-1)} - L_{i,k}^2$$

**Step 5: The Truncation Condition**
If the maximum value of the newly updated diagonals drops below the numerical tolerance (e.g., $10^{-7}$), the remaining space is fully spanned by the active basis, and the algorithm terminates:
$$\max(\mathbf{d}^{(k)}) < \epsilon_{\text{tol}}$$