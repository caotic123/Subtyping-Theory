# Open signatures: non-bidirectional reformulation

Compile `open-signatures.tex` with `latexmk -pdf open-signatures.tex`
(or run `pdflatex open-signatures.tex` twice).

## Main change

All source expressions use the single declarative judgment
`Gamma |- e : A ~> t`, typeset with `\elab{e}{A}{t}`.
The core judgment `Gamma |- t : A` remains unchanged.

- Former synthesis and checking rules use the same relation.
- `el_conv` combines both conversion rules and preserves the core term.
- `el_sub` permits subsumption after any source derivation and inserts `c t`.
- Applications and dependent pairs substitute elaborated core terms in types.
- Constructors, scrutinees, and live clause bodies use the unified relation.
- Annotation is a type constraint, not a mode switch.
- All core and coercion rules, including the fixed recursive `G` in
  `su_close`, remain unchanged.

## Rule mapping

| Original label | New label |
| --- | --- |
| es_var | el_var |
| es_ann | el_ann |
| es_app | el_app |
| ec_lam | el_lam |
| ec_pair | el_pair |
| ec_core | el_core |
| ec_conversion | el_conv |
| ec_target_conversion | el_conv |
| ec_subsumption | el_sub |
| es_signature | el_signature |
| es_close | el_close |
| ec_constructor | el_constructor |
| es_case | el_case |

`cases_live` retains its label but its source-typing premise is changed.
The names of the new source rules are proposed names, not claims about the
contents of any Rocq implementation.

## Scope and metatheory

This is a declarative specification, not a deterministic type-inference
algorithm. Original derivations embed after erasing modes and translating
clause derivations, but the reverse correspondence is not asserted.
Unrestricted composition allows previously unavailable derivations.

The original soundness statements are combined as elaboration soundness
and remain unproved. No Rocq `.v` files were supplied or generated.
The original notes' counts and proof-status reports are explicitly attributed
to those notes rather than presented as independently verified here.

The former checking-coherence statement was NOT blindly renamed: Section 13
contains a concrete counterexample to unconditional elaboration coherence
using an unannotated core value with two source types and different retagging
coercions into a common target. Coercion coherence for fixed endpoints is a
separate statement and retains its original conjectural status.

## Files

The original six-file organization is retained. `open-signatures-core.tex`
is byte-for-byte unchanged. The references retain the original bibliography,
including local links that require the author's original workspace.
The companion `changes.patch` shows the complete changes from the uploads.
