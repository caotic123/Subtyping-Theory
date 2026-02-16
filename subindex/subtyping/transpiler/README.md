# siglp-transpiler

Minimal transpiler from a simplified LP surface language to LambdaPi code
that targets the signature encoding in `subindex/subtyping/encode.lp`.
The parser is implemented with `parsec`.

## Supported syntax

- `Inductive T ...;`
  - Registers `T` for signature shorthand.
  - Extracts constructor names from the inductive block.
  - Infers index shape from the inductive type head (e.g. `Set → nat → TYPE`).
  - Also accepts constrained index domains in the head, e.g. `Set → {nat | 0 | 1} → TYPE`.
    - Current behavior: this is used to infer the base index kind (`nat` here), while
      constructor alternatives inside `{...}` are parsed but not yet used to refine delta patterns.
  - The transpiler injects:
    - `T__sig_target : Set` with rule to `TLabel`
    - `T__sig_indexes : indexes_type` inferred from type parameters
    - `T__sig_pattern_any : index_pattern_list` (all-variable pattern)
    - `T__sig_delta : delta_tel` as `mk_delta_tel T__sig_indexes T__sig_pattern_any`
    - `T__sig_phi_all : constructor_list` from constructor names in the inductive block
- `Def f : A := t;`
  - Translates to:
    - `symbol f : A;`
    - `rule f ↪ t;`
- Signature shorthand:
  - `{T :: c1 | c2 | ...}` where `T` was declared by `Inductive T ...;`
  - `{T | c1 | c2 | ...}` is also accepted as equivalent shorthand.
  - For indexed inductives, constructor-filtered signatures require explicit telescope arguments:
    - accepted: `{vec nat v | cons}`
    - rejected: `{vec | cons}`
  - Indexed form is supported for declared inductives: `{vec a v}` or `{vec a (+1 v) | cons}`.
    - This expands to `mk_delta_tel vec__sig_indexes <pattern>` automatically.
    - Current implementation refines nat-like indices (e.g. `v`, `0`, `+1 v` / `S v`); polymorphic `Set` indices stay generalized (`index_pat_var`).
  - Nat subset syntax `{nat | ...}` is accepted as a surface shorthand:
    - In type/term positions it currently lowers to `nat`.
    - In `match ... return ...` it lowers to `natLabel` (required by `signature_case`).
  - Translates to `signature_term T__sig_target T__sig_delta <constructor_list_expr>`.
  - `{T}` (no `::`) uses all constructors from `Inductive T ...`.
  - `{T @ g :: c1 | c2 | ...}`
  - Also accepts `{T g :: ...}` when `T` and `g` are simple space-separated terms.
  - Translates to `signature_term T g <constructor_list_expr>`.
  - Constructor names without suffix are normalized: `cons` -> `consLabel`.
- Signature-only match:
  - `match m as {T @ g :: c1 | ...} return Q with | c x => n ... end`
  - Translates to `signature_case ... (case_branches_cons ...)`.
  - Any non-signature `match` form is rejected by the transpiler.
- Any other top-level statement is passed through.

## Not supported

- Direct `symbol` / `rule` declarations in source (rejected).
- Full automatic backend generation for arbitrary constructor labels (current backend knows `emptyLabel`/`consLabel`).
- Full LP grammar parsing; this is a soft parser intended for the subset above.

## Build

```bash
cd subindex/subtyping/transpiler
cabal build
```

## Run

```bash
cabal run siglp-transpiler -- examples/sample.siglp examples/sample.lp
```
