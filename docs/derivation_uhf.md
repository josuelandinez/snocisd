---
title: "Extension of CISD Compression to Compact Non-Orthogonal Determinant Expansions: The UHF Case"
author: "Documentation"
geometry: margin=1in
output: pdf_document
---

# Extension of CISD Compression to Compact Non-Orthogonal Determinant Expansions: The UHF Case

### 1. Motivation

The "Compressed CISD" method [1] offers a rigorous formalism for representing the exact CISD wavefunction using a compact basis of non-orthogonal Slater determinants (NODs). The original derivation, however, relies strictly on the **Restricted Hartree-Fock (RHF)** approximation, assuming spatial symmetry between $\alpha$ and $\beta$ orbitals ($\phi_i^\alpha = \phi_i^\beta$).

This assumption breaks down in:

1.  **Open-Shell Systems** ($N_\alpha \neq N_\beta$), where the $\alpha$ and $\beta$ Hilbert spaces have different dimensions.
2.  **Spin-Polarized Systems** (e.g., bond breaking, antiferromagnetism), where $\phi_i^\alpha \neq \phi_i^\beta$. Even if electron counts are equal, the relaxation pathways for $\alpha$ and $\beta$ electrons differ.

In the general case, the "relaxation terms" ($\hat{U}^2$ and $\hat{V}^2$) that appear during finite difference approximations become "noise" rather than signal. We present a **General UHF Extension** that rigorously treats distinct $\alpha$ and $\beta$ spin manifolds using a **4-Point Stencil** to systematically eliminate these unphysical relaxation errors.

---

### 2. Theoretical Framework

The exact CISD wavefunction in the UHF basis is defined as:

$$
|\Psi_{\text{CISD}}\rangle = c_0 |\Phi_0\rangle + \hat{T}_1 |\Phi_0\rangle + \hat{T}_2 |\Phi_0\rangle \tag{1}
$$

where $\hat{T}_1$ and $\hat{T}_2$ are cluster operators defined over spin-orbitals.

#### A. Single Excitations ($\hat{T}_1$)

The single excitation operator is strictly separable by spin:

$$
\hat{T}_1 = \hat{T}_{1\alpha} + \hat{T}_{1\beta} \tag{2}
$$

Using the **Thouless Theorem**, we define a determinant $|\Phi(Z)\rangle = e^{\hat{Z}} |\Phi_0\rangle$. A linear single excitation is represented exactly by the finite difference of two determinants in the limit $\delta \to 0$:

$$
\hat{T}_1 |\Phi_0\rangle = \lim_{\delta \to 0} \frac{1}{2\delta} \left( |\Phi(+\delta \hat{T}_1)\rangle - |\Phi(-\delta \hat{T}_1)\rangle \right) \tag{3}
$$

* **Implementation:** Requires **2 determinants** (+ Reference).

#### B. Double Excitations ($\hat{T}_2$)

Due to spin orthogonality, the UHF double excitation operator splits into three distinct blocks:

$$
\hat{T}_2 = \hat{T}_{\alpha\alpha} + \hat{T}_{\beta\beta} + \hat{T}_{\alpha\beta} \tag{4}
$$

To compress these terms, we perform block-specific decompositions (SVD) and reconstruct the operators using finite difference stencils.

*(See Appendix A for the detailed derivation of why the 4-Point Stencil is required for these blocks).*

---

### 3. The Mixed-Spin Block ($\alpha\beta$)

This term describes the correlation between distinguishable spins.

$$
\hat{T}_{\alpha\beta} = \sum_{ia, jb} t_{ia, jb} \, \hat{a}_a^\dagger \hat{a}_i \, \hat{b}_b^\dagger \hat{b}_j \tag{5}
$$

The amplitude matrix $\mathbf{T}_{\alpha\beta}$ is **Rectangular** ($N_{\text{ex}}^\alpha \times N_{\text{ex}}^\beta$).

#### Step 1: Singular Value Decomposition (SVD)

We decompose the rectangular matrix of amplitudes ($u_k$ and $v_k$ are vectors):

$$
\mathbf{T}_{\alpha\beta} = \sum_{k=1}^{R} \sigma_k \mathbf{u}_k \mathbf{v}_k^T \tag{6}
$$

This yields coupled operators $\hat{Q}_k = \sigma_k \hat{U}_k \hat{V}_k$.

#### Step 2: The 4-Point Stencil (Balanced Quartets)

To isolate the correlation product $\hat{U}\hat{V}$ without contamination from single-spin relaxation terms ($\hat{U}^2, \hat{V}^2$), we employ a 4-point stencil on the sum and difference operators $\hat{Q}_+ = \hat{U} + \hat{V}$ and $\hat{Q}_- = \hat{U} - \hat{V}$:

$$
\hat{U}\hat{V} |\Phi_0\rangle \approx \frac{1}{4\delta^2} \left[ \sum_{\sigma=\pm} |\Phi(\sigma \delta Q_+)\rangle - \sum_{\sigma=\pm} |\Phi(\sigma \delta Q_-)\rangle \right] \tag{7}
$$

---

### 4. The Same-Spin Blocks ($\alpha\alpha$ / $\beta\beta$)

The amplitude tensor must be **Antisymmetric** due to the Pauli exclusion principle ($t_{ij}^{ab} = -t_{ji}^{ab}$):

$$
\mathbf{T}_{\alpha\alpha}^T = -\mathbf{T}_{\alpha\alpha} \tag{8}
$$

#### Step 1: Skew-Symmetric Decomposition

We use the canonical block-diagonal decomposition for skew-symmetric matrices, yielding pairs of vectors $\mathbf{l}_k, \mathbf{r}_k$ corresponding to the singular values $\sigma_k$:

$$
\mathbf{T}_{\alpha\alpha} = \sum_{k=1}^{R/2} \sigma_k (\mathbf{l}_k \mathbf{r}_k^T - \mathbf{r}_k \mathbf{l}_k^T) \tag{9}
$$

#### Step 2: Representation via Identity

Using the 4-Point Stencil on $Q_+ = L+R$ and $Q_- = L-R$, we extract the product:

$$
\hat{L}\hat{R} |\Phi_0\rangle \approx \frac{1}{4\delta^2} \left[ \sum_{\sigma=\pm} |\Phi(\sigma \delta Q_+)\rangle - \sum_{\sigma=\pm} |\Phi(\sigma \delta Q_-)\rangle \right] \tag{10}
$$

The antisymmetry ($\hat{L}\hat{R} - \hat{R}\hat{L}$) is automatically enforced by the Slater determinant structure acting on the vacuum state.

---

### 5. Summary and Cost Analysis

The final compact wavefunction is a linear combination of non-orthogonal determinants:

$$
|\Psi_{\text{NOCI}}\rangle = c_0 |\Phi_0\rangle + \sum_{k} x_k |\Phi_k\rangle
$$

| Component | Equation | Determinants Added |
| :--- | :--- | :--- |
| **Reference** | $|\Phi_0\rangle$ | 1 |
| **Singles** | Eq. 3 | 2 |
| **Doubles ($\alpha\beta$)** | Eq. 7 | $4 \times \text{Rank}(\alpha\beta)$ |
| **Doubles ($\alpha\alpha$)** | Eq. 10 | $4 \times \text{Rank}(\alpha\alpha)$ |
| **Doubles ($\beta\beta$)** | Eq. 10 | $4 \times \text{Rank}(\beta\beta)$ |

---

### 6. Energy Computation via Generalized Eigenvalue Problem

To verify the accuracy of the compressed wavefunction, we must calculate its variational energy. Since the basis determinants $|\Phi_k\rangle$ are non-orthogonal, this requires solving a **Generalized Eigenvalue Problem**.

#### A. Constructing the Matrices

We define the overlap matrix $\mathbf{S}$ and Hamiltonian matrix $\mathbf{H}$ in the basis of the generated determinants $\{|\Phi_k\rangle\}$:

**1. Overlap Matrix ($\mathbf{S}$)**
$$
S_{uv} = \langle \Phi_u | \Phi_v \rangle = \det(\mathbf{C}_u^\dagger \mathbf{C}_v) \tag{11}
$$
where $\mathbf{C}_u$ is the coefficient matrix of determinant $u$.

**2. Hamiltonian Matrix ($\mathbf{H}$)**
$$
H_{uv} = \langle \Phi_u | \hat{H} | \Phi_v \rangle \tag{12}
$$
This is computed using the **Generalized Slater-Condon Rules** (Löwdin formula) which accounts for non-orthogonality via cofactors of the overlap matrix.

#### B. Variational Solution

The energy $E$ and coefficients $\mathbf{c}$ of the compressed wavefunction $|\Psi_{NOCI}\rangle = \sum_k c_k |\Phi_k\rangle$ are found by solving:

$$
\mathbf{H} \mathbf{c} = E \mathbf{S} \mathbf{c} \tag{13}
$$

The lowest eigenvalue $E_0$ corresponds to the variational energy of the compressed state.

* **Verification:** If the compression is exact, $E_0 \approx E_{\text{CISD}}$.
* **Bonus:** Since $|\Phi_k\rangle = e^{\hat{Z}_k}|\Phi_0\rangle$, the basis naturally includes disconnected higher-order excitations (e.g., $\frac{1}{2}\hat{T}_2^2$). Thus, it is possible to recover energies slightly **lower** than linear CISD (capturing approximate CCSD correlation).

---

# Appendices: Detailed Derivations

### Appendix A: Explicit Derivation of 3-Point vs. 4-Point Stencils

**Context:** In the restricted case ($N_\alpha = N_\beta$), "relaxation terms" ($\hat{U}^2$ and $\hat{V}^2$) are harmless or even helpful because symmetry allows them to be absorbed into the single coefficient. In the general case, they become "noise" that needs to be cancelled.

Here is the explicit, step-by-step derivation for the mixed-spin block ($\hat{U}_\alpha \hat{V}_\beta$), showing exactly why the determinant counts differ.



#### 1. The Goal

We want to represent the correlation operator:

$$
\hat{O}_{corr} = \hat{U}_\alpha \hat{V}_\beta
$$

using Thouless determinants $|\Phi(\dots)\rangle$.

* $\hat{U}_\alpha$: Pure $\alpha$ excitation (behaves like $T_1^\alpha$).
* $\hat{V}_\beta$: Pure $\beta$ excitation (behaves like $T_1^\beta$).
* They commute: $[\hat{U}, \hat{V}] = 0$.

#### 2. Derivation of the 3-Point Stencil
*Used in the original paper for Restricted Singlets.*

We define a combined operator $\hat{Q} = \hat{U} + \hat{V}$. We construct three determinants:

1.  **Reference:** $|\Phi_0\rangle$ (Scaling factor $\delta = 0$)
2.  **Positive Step:** $|\Phi(+\delta \hat{Q})\rangle$
3.  **Negative Step:** $|\Phi(-\delta \hat{Q})\rangle$

**Step A: Taylor Expansion**

The Thouless determinant can be expanded as an exponential:

$$
|\Phi(\delta \hat{Q})\rangle = e^{\delta(\hat{U}+\hat{V})} |\Phi_0\rangle
$$

Using the Taylor series $e^x = 1 + x + \frac{x^2}{2} + \dots$:

$$
|\Phi(+\delta \hat{Q})\rangle = \left[ 1 + \delta(\hat{U}+\hat{V}) + \frac{\delta^2}{2}(\hat{U}+\hat{V})^2 + O(\delta^3) \right] |\Phi_0\rangle
$$

$$
|\Phi(-\delta \hat{Q})\rangle = \left[ 1 - \delta(\hat{U}+\hat{V}) + \frac{\delta^2}{2}(\hat{U}+\hat{V})^2 - O(\delta^3) \right] |\Phi_0\rangle
$$

**Step B: The Central Difference Sum**

Add the two equations together:

$$
|\Phi(+\delta)\rangle + |\Phi(-\delta)\rangle = 2|\Phi_0\rangle + \delta^2 (\hat{U}+\hat{V})^2 |\Phi_0\rangle + O(\delta^4)
$$

Expand the square term $(\hat{U}+\hat{V})^2 = \hat{U}^2 + \hat{V}^2 + 2\hat{U}\hat{V}$:

$$
|\Phi(+\delta)\rangle + |\Phi(-\delta)\rangle - 2|\Phi_0\rangle = \delta^2 (\hat{U}^2 + \hat{V}^2 + \mathbf{2\hat{U}\hat{V}}) |\Phi_0\rangle
$$

**Step C: The Result**

Solving for our target $\hat{U}\hat{V}$:

$$
\mathbf{\hat{U}\hat{V}} |\Phi_0\rangle \approx \underbrace{\frac{|\Phi(+\delta)\rangle + |\Phi(-\delta)\rangle - 2|\Phi_0\rangle}{2\delta^2}}_{\text{Numerical 2nd Derivative}} - \underbrace{\frac{1}{2}(\hat{U}^2 + \hat{V}^2)|\Phi_0\rangle}_{\text{Relaxation Error}}
$$

* **The Issue:** We are left with $-\frac{1}{2}(\hat{U}^2 + \hat{V}^2)$. In the general case ($\hat{U} \neq \hat{V}$), we cannot separate this "error" from the signal.

#### 3. Derivation of the 4-Point Stencil
*Used in the General Code ("Balanced Quartets").*

We need to cancel the $\hat{U}^2 + \hat{V}^2$ terms. We use a Mixed Partial Derivative approach involving two axes of rotation.

Define two operators:

1.  $\hat{Q}_+ = \hat{U} + \hat{V}$
2.  $\hat{Q}_- = \hat{U} - \hat{V}$

We construct four determinants (plus reference):

* $|\Phi(+\delta Q_+)\rangle, |\Phi(-\delta Q_+)\rangle$
* $|\Phi(+\delta Q_-)\rangle, |\Phi(-\delta Q_-)\rangle$

**Step A: Expansion of $Q_+$ (Sum)**

From the 3-point derivation above, we know:

$$
\text{Sum}_+ = |\Phi(+\delta Q_+)\rangle + |\Phi(-\delta Q_+)\rangle \approx 2|\Phi_0\rangle + \delta^2 (\hat{U}^2 + \hat{V}^2 + \mathbf{2\hat{U}\hat{V}})
$$

**Step B: Expansion of $Q_-$ (Difference)**

Now expand $(\hat{U}-\hat{V})^2 = \hat{U}^2 + \hat{V}^2 - 2\hat{U}\hat{V}$:

$$
\text{Sum}_- = |\Phi(+\delta Q_-)\rangle + |\Phi(-\delta Q_-)\rangle \approx 2|\Phi_0\rangle + \delta^2 (\hat{U}^2 + \hat{V}^2 - \mathbf{2\hat{U}\hat{V}})
$$

**Step C: The Subtraction (Cancellation)**

Subtract $\text{Sum}_-$ from $\text{Sum}_+$:

$$
\text{Sum}_+ - \text{Sum}_- = \delta^2 [ (\hat{U}^2 + \hat{V}^2 + 2\hat{U}\hat{V}) - (\hat{U}^2 + \hat{V}^2 - 2\hat{U}\hat{V}) ]
$$

The squared terms $\hat{U}^2$ and $\hat{V}^2$ cancel exactly:

$$
\text{Sum}_+ - \text{Sum}_- = \delta^2 (4\hat{U}\hat{V})
$$

**Step D: The Result**

$$
\mathbf{\hat{U}\hat{V}} |\Phi_0\rangle \approx \frac{1}{4\delta^2} \left[ \Big(|\Phi(Q_+)\rangle + |\Phi(-Q_+)\rangle\Big) - \Big(|\Phi(Q_-)\rangle + |\Phi(-Q_-)\rangle\Big) \right]
$$

* **The Benefit:** Pure correlation term. Zero relaxation error.

---

### Appendix B: Formal Proof of Skew-Symmetric Decomposition

**Theorem:** Any real skew-symmetric matrix $\mathbf{A}$ ($\mathbf{A}^T = -\mathbf{A}$) can be block-diagonalized.

**Proof:**

1.  Since $\mathbf{A}$ is real skew-symmetric, $i\mathbf{A}$ is Hermitian. Eigenvalues are purely imaginary pairs $\pm i\sigma_k$.
2.  Let $\mathbf{z} = \mathbf{x} + i\mathbf{y}$ be an eigenvector of $\mathbf{A}$ with eigenvalue $i\sigma$.
    $$
    \mathbf{A}(\mathbf{x} + i\mathbf{y}) = i\sigma(\mathbf{x} + i\mathbf{y}) = -\sigma \mathbf{y} + i\sigma \mathbf{x}
    $$
3.  Separating components:
    $$
    \mathbf{A}\mathbf{x} = -\sigma \mathbf{y}, \quad \mathbf{A}\mathbf{y} = \sigma \mathbf{x} \tag{B.1}
    $$
4.  In the subspace $\{\mathbf{x}, \mathbf{y}\}$, $\mathbf{A}$ acts as $\begin{pmatrix} 0 & \sigma \\ -\sigma & 0 \end{pmatrix}$. Normalizing $\mathbf{x} \to \mathbf{L}$ and $\mathbf{y} \to \mathbf{R}$:
    $$
    \mathbf{A} = \sum_k \sigma_k (\mathbf{L}_k \mathbf{R}_k^T - \mathbf{R}_k \mathbf{L}_k^T) \tag{B.2}
    $$

**Conclusion:** This justifies Eq. 9 for same-spin blocks.

---

### Appendix C: Construction of Non-Orthogonal Determinants

**Context:** This section details how the abstract Thouless rotation operator $e^{\hat{Z}}$ is explicitly converted into the coefficient matrix used in the code.

**Step 1: The Operator Definition**

The Thouless operator $\hat{Z}$ promotes electrons from occupied orbitals $i$ to virtual orbitals $a$:

$$
\hat{Z} = \sum_{ai} Z_{ai} \hat{a}_a^\dagger \hat{a}_i
$$

We seek the new orbital set $\tilde{\phi}$ such that $|\Phi(Z)\rangle = \det(\tilde{\phi}_1 \dots \tilde{\phi}_N) = e^{\hat{Z}} |\Phi_0\rangle$.

**Step 2: The Nilpotency Property**

Consider the action of $e^{\hat{Z}}$ on a single occupied orbital $\phi_i$:

$$
\tilde{\phi}_i = e^{\hat{Z}} \phi_i = \left( 1 + \hat{Z} + \frac{1}{2}\hat{Z}^2 + \dots \right) \phi_i
$$

* **First Term:** $1 \cdot \phi_i = \phi_i$ (Original occupied orbital).
* **Second Term:** $\hat{Z} \phi_i = \sum_a Z_{ai} \phi_a$ (Excitation to virtual space).
* **Third Term:** $\hat{Z}^2 \phi_i = \hat{Z} (\sum_a Z_{ai} \phi_a)$. The operator $\hat{Z}$ attempts to annihilate an occupied state ($\hat{a}_i$), but the state $\sum Z_{ai} \phi_a$ is entirely virtual. Thus, $\hat{Z}^2 \phi_i = 0$.

Since $\hat{Z}^n \phi_i = 0$ for $n \ge 2$, the series truncates exactly after the first order:

$$
\tilde{\phi}_i = \phi_i + \sum_a Z_{ai} \phi_a \tag{C.1}
$$

**Step 3: Matrix Construction**

In the code, we represent orbitals as column vectors in the full basis (Occupied $\oplus$ Virtual).

* The original occupied orbital $\phi_i$ corresponds to the unit vector $\mathbf{e}_i$ in the occupied block.
* The correction term $\sum Z_{ai} \phi_a$ corresponds to the vector $\mathbf{z}_i$ in the virtual block.

Stacking these vectors for all $N$ electrons yields the coefficient matrix:

$$
\mathbf{C}(Z) = \begin{pmatrix} \mathbf{I}_{occ} \\ \mathbf{Z}_{vir \times occ} \end{pmatrix} \tag{C.2}
$$

This matrix $\mathbf{C}(Z)$ is what is passed to the NOCI solver. It represents the Thouless-rotated determinant exactly.

---

# Bibliography

[1] Sun, C., Gao, F., & Scuseria G. E.. (2024). *Selected non-orthogonal configuration interaction with compressed single and double excitations*. arXiv preprint arXiv:2403.02350.