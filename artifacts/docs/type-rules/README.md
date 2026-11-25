# PomPom typing rules

> Copied from [PomPom-Language](https://github.com/caotic123/PomPom-Language) `docs/type-rules`
> at commit `046e4b9` (branch `first-class-datatypes`, 2026-10-03), the latest at copy time.
> Only local paths in `open-signatures-references.tex` were adapted to this repository;
> links to `progress/` and `proof-archive/` point to that commit on GitHub.

An [explicitly annotated core repair](https://github.com/caotic123/PomPom-Language/blob/046e4b9/progress/annotated/README.md) is
under construction following the September 30 design decision. The PDF below
still documents the existing named calculus and its checked counterexample;
it will be migrated when the annotated preservation proof is complete.

## Open signatures and parameterized closure

The current rule reference is
[open-signatures.pdf](open-signatures.pdf), built from
[open-signatures.tex](open-signatures.tex) and its included files. It keeps
prose short while displaying the current context, typing, name-resolution,
and elaboration rules, with explicit formation premises and searchable Rocq
rule names. Function cumulativity is contravariant in domains and covariant
in codomains, recursively through dependent functions; both function types
must be well formed. The reference includes the eta regression examples,
structural computation equations, concrete list/nonempty-list/tree
instantiations, dependent function coercions, and a bibliography.

Forty-six original claims have proofs without axiom dependencies, including
coercion and checking coherence, and full preservation is refuted. No
conjectures remain. The PDF and its
LaTeX sources distinguish these statuses; see the
[current proof status](https://github.com/caotic123/PomPom-Language/blob/046e4b9/progress/PROOF_STATUS.md) for exact dependencies.
Named binders use [stable numeric IDs and an alpha setoid](https://github.com/caotic123/PomPom-Language/blob/046e4b9/progress/STABLE_IDS.md).
The archived eta counterexample concerns the
[pre-repair rules](https://github.com/caotic123/PomPom-Language/blob/046e4b9/proof-archive/2026-09-27-before-function-cumulativity/README.md).
The active [regressions](https://github.com/caotic123/PomPom-Language/blob/046e4b9/progress/OpenSignaturesEtaRegression.v) check
both function variance directions and a nested dependent function, without
assuming full preservation. Execution preservation and elaboration soundness
are proved. Typed computation/eta factorization removes the earlier
full-preservation dependency from description views. The separate
[polymorphic application counterexample](https://github.com/caotic123/PomPom-Language/blob/046e4b9/progress/OpenSignaturesEtaPolymorphism.v)
refutes full preservation even with function cumulativity. Its false axiom
was removed, with the exact proposition retained for reference. The two
coherence claims are proved through a binary relational model. Full beta/eta strong normalization
is proved for every typing derivation, using the cumulative semantic model,
positive fixed points, computability of every primitive operator, and the
full fundamental typing theorem. Semantic environments and substitution
reflection transfer the result to the original named calculus. Consistency
and all six formerly conditional claims are now closed without assumptions.
The current source snapshot is recorded in
[open-signatures-source-sha256.txt](open-signatures-source-sha256.txt).

| LaTeX source | Contents |
| --- | --- |
| [open-signatures-core.tex](open-signatures-core.tex) | Contexts, universes, Π/Σ, enumerations, descriptions, `iAll`, `hyps`, reference μᴵ, and `close`. |
| [open-signatures-elaboration.tex](open-signatures-elaboration.tex) | Rows, dead witnesses, row handlers, description/type coercions, source checking and synthesis, constructors, and cases. |
| [open-signatures-examples.tex](open-signatures-examples.tex) | Lists, nonempty lists, constructor reuse, compact/padded conversions, observers, and instantiated induction hypotheses. |
| [open-signatures-theorems.tex](open-signatures-theorems.tex) | Selected theorem statements, current proof status, and the source map. |
| [open-signatures-references.tex](open-signatures-references.tex) | Paper citations and attribution of inherited constructions versus proposed rules. |

Build it from the repository root:

```sh
make -C artifacts/docs/type-rules
```

It uses `latexmk` and pdfLaTeX; `make -C artifacts/docs/type-rules clean`
removes auxiliary files while retaining the PDF.
