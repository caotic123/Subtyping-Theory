# Extracted Type System Rules and Theorems

## Type System Rules

### Rule 1: Signature Formation
```
Γ ⊢ T Δ* : Type    Φ = (C₁, ...Cₙ)    (C₁ ... Cₙ) ⊆ 𝒞_all
∀i, i ≤ n, Γ ⊢ Cᵢ : Δ → T Δ'*
∀j, j ≤ |Δ*|, AGAINST(Δ*ⱼ, Δ'*ⱼ)
─────────────────────────────────────
Γ ⊢ {T Δ* :: Φ} : Type
```

### Rule 2: Signature Subsumption
```
Γ ⊢ T : Type
─────────────────────────────────
Γ ⊢ {T :: Φ'} ⊑ T
```

### Rule 3: Constructor Typing
```
Γ ⊢ T : Type    Φ = (C₁, ...Cₙ)
Γ ⊢ C : Δ → T    Cᵢ ∈ Φ, ∀i, 1 ≤ i ≤ n
Cᵢ Δ_βη, C ∈ 𝒞_all
─────────────────────────────────
Γ ⊢ Cᵢ Δ : {T :: Φ}
```

### Rule 4: Signature Subtyping
```
T =_βη T'    Φ' ⊆ Φ
─────────────────────────────────
Γ ⊢ {T' :: Φ'} ⊑ {T :: Φ}
```

### Rule 5: Function Subtyping
```
Γ ⊢ T : Type    Γ ⊢ F : Δ → A
Γ ⊢ A ⊑ {T :: Φ'}
F Δ_βη, F ∉ 𝒞_all
─────────────────────────────────
Γ ⊢ F Δ : {T :: Φ'}
```

### Rule 6: Pi Type Subtyping (Contravariant/Covariant)
```
Γ ⊢ A' ⊑ A    Γ, x:A' ⊢ B ⊑ B'
─────────────────────────────────
Γ ⊢ (x : A) → B ⊑ (x : A') → B'
```

### Rule 7: Case Expression (Pattern Matching)
```
Γ ⊢ T Δ* : Type    Γ ⊢ Q : Type
Φ = (C₁ : Δ₁ → T Δ*₁, ..., Cₙ : Δₙ → T Δ*ₙ)
Γ ⊢ M : {T Δ* :: Φ}
Γ ⊢ ∀i ≤ |Φ|, Nᵢ : Δᵢ → Q
─────────────────────────────────
Γ ⊢ case M of Q {Cᵢ Δᵢ ⇒ Nᵢ Δᵢ, ...} : Q
```

### Subsumption Rule
```
Γ ⊢ t : A    Γ ⊢ A ⊑ B
─────────────────────────────────
Γ ⊢ t : B
```

## AGAINST Rules (Index Compatibility)

```
─────────────────────────────────
AGAINST(∅, ∅)
```

```
AGAINST(Δ, Δ')    v is Var
─────────────────────────────────
AGAINST(v ... Δ, c ... Δ')
```

```
AGAINST(Δ, Δ')    (c, c') ⊆ 𝒞_all    c Δᶜ =_α c' Δᶜ'
─────────────────────────────────
AGAINST((c Δᶜ) ... Δ, (c' Δᶜ') ... Δ')
```

## Reduction Rules

### Application Reduction
```
L → L'
─────────────────────────────────
L · M → L' · M
(ξ-·app₁)
```

```
M → M'
─────────────────────────────────
V · M → V · M'
(ξ-·app₂)
```

### Beta Reduction
```
─────────────────────────────────
(λx ⇒ N) · V → N[x := V]
(ξ-β)
```

### Case Reduction
```
v → v'
─────────────────────────────────
case v of Q {Cᵢ Δᵢ ⇒ Nᵢ Δᵢ, ...} → case v' of Q {Cᵢ Δᵢ ⇒ Nᵢ Δᵢ, ...}
(ξ-case)
```

```
─────────────────────────────────
case (Cᵢ Δ') of Q {Cᵢ Δᵢ ⇒ Nᵢ Δᵢ, ...} → Nᵢ Δ'
(ξ-case')
```

## Theorems

### Theorem 1: Isomorphism with Sigma Types
Let T be an inductive family with constructor set C, and let S ⊆ C. Define a predicate D : T → Type such that for every constructor application C' Δ,

```
D(C' Δ) ≡ {⊤  if C' ∈ S,
           ⊥  if C' ∉ S.
```

Then `{T :: S} ≃ Σ(x : T), D(x)`.

**Proof sketch:**
We define mutually inverse functions between {T :: S} and Σ(x : T), D(x).
- Given x : {T :: S}, the underlying term of x is built only from constructors in S (by Rule 7). Therefore D(x) is provable, and we obtain f(x) ≔ (x, dₓ) : Σ(y : T), D(y).
- Conversely, given (y, d) : Σ(x : T), D(x), the proof d : D(y) guarantees that y was built using only constructors from S. Thus y inhabits {T :: S}, and we define g(y,d) ≔ y : {T :: S}.

By construction we have f(g(y,d)) = (y,d) and g(f(x)) = x.

### Theorem 2: Progress
If `· ⊢ M : {T :: Φ}`, then either M is a value or there exists M' such that M → M'.

**Proof sketch:**
By structural induction on M and case analysis:
- Application M = M₁ · M₂: If either is not a value, use IH and congruence rules. If both are values and M₁ is a lambda, perform β-reduction.
- Case expression: If scrutinee v is not a value, use ξ-case. If v is a value, by Rule 7 there exists a matching pattern Cᵢ, so use ξ-case'.
- Other forms: Canonical forms analysis shows closed terms of signature type are either values or reduce.

### Theorem 3: Preservation (Subject Reduction)
If `Γ ⊢ M : R` and `M → M'`, then `Γ ⊢ M' : R`.

**Proof sketch:**
By induction on the evaluation derivation:
- β-reduction: Use substitution lemma.
- ξ-app₁, ξ-app₂: IH preserves subterm type, reapply typing rule.
- ξ-case: IH gives `Γ ⊢ v' : {T :: Φ}`, Rule 7 re-establishes `Γ ⊢ M' : Q`.
- ξ-case': Rule 7 requires `Γ ⊢ Nᵢ : Δᵢ → Q`, instantiating with Δ' gives type Q.

## Key Properties

1. **Constructor Subset Subtyping**: If Φ' ⊆ Φ, then {T :: Φ'} ⊑ {T :: Φ}

2. **Phantom Type Erasure**: Signature information {T :: Φ} can be erased to T (Rule 2)

3. **Coverage Checking**: Pattern matching on {T :: Φ} only requires cases for constructors in Φ (Rule 7)

4. **Type Safety**: The system satisfies both Progress and Preservation theorems
