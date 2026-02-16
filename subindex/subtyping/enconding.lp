require open subtyping.encode;

// Common constructor-list aliases used by the example signatures below.
symbol phi_vec_all : constructor_list;
rule phi_vec_all ↪
  constructor_list_cons emptyLabel (constructor_list_cons consLabel constructor_list_nil);

symbol phi_vec_cons_only : constructor_list;
rule phi_vec_cons_only ↪
  constructor_list_cons consLabel constructor_list_nil;

symbol phi_vec_empty_only : constructor_list;
rule phi_vec_empty_only ↪
  constructor_list_cons emptyLabel constructor_list_nil;

symbol vec_any_pattern : index_pattern_list;
rule vec_any_pattern ↪
  index_pattern_cons index_pat_var
    (index_pattern_cons index_pat_var index_pattern_nil);

symbol vec_indexes : indexes_type;
rule vec_indexes ↪
  index_poly_succ (index_dependent_lift_succ natLabel index_null);

symbol vec_delta_any : delta_tel;
rule vec_delta_any ↪
  mk_delta_tel vec_indexes vec_any_pattern;

// Example encoding of vec s 0 (for any s): first index is variable, second is 0.
symbol vec_zero_pattern : index_pattern_list;
rule vec_zero_pattern ↪
  index_pattern_cons index_pat_var
    (index_pattern_cons
      (index_pat_term (index_term_ctor _0 index_term_nil))
      index_pattern_nil);

symbol vec_delta_zero : delta_tel;
rule vec_delta_zero ↪
  mk_delta_tel vec_indexes vec_zero_pattern;

// Example encoding of vec s (S n) (non-empty vectors).
symbol vec_nonempty_pattern : index_pattern_list;
rule vec_nonempty_pattern ↪
  index_pattern_cons index_pat_var
    (index_pattern_cons
      (index_pat_term
        (index_term_ctor
          (+1 _0)
          (index_term_cons index_term_var index_term_nil)))
      index_pattern_nil);

symbol vec_delta_nonempty : delta_tel;
rule vec_delta_nonempty ↪
  mk_delta_tel vec_indexes vec_nonempty_pattern;

symbol vec_signature : signature vecLabel vec_delta_any phi_vec_all;
rule vec_signature ↪
  signature_cons
    vecLabel
    vec_delta_any
    emptyLabel
    (constructor_list_cons consLabel constructor_list_nil)
    top
    (signature_cons vecLabel vec_delta_any consLabel constructor_list_nil top (signature_nil vecLabel vec_delta_any));

// Example signature for vec s 0: only empty is admissible.
symbol vec_zero_signature : signature vecLabel vec_delta_zero phi_vec_empty_only;
rule vec_zero_signature ↪
  signature_cons
    vecLabel
    vec_delta_zero
    emptyLabel
    constructor_list_nil
    top
    (signature_nil vecLabel vec_delta_zero);

// Example signature for non-empty vectors: only cons is admissible.
symbol vec_nonempty_signature : signature vecLabel vec_delta_nonempty phi_vec_cons_only;
rule vec_nonempty_signature ↪
  signature_cons
    vecLabel
    vec_delta_nonempty
    consLabel
    constructor_list_nil
    top
    (signature_nil vecLabel vec_delta_nonempty);

// Cons-only signature at the "any index" telescope (used for coercion demos).
symbol vec_cons_any_signature : signature vecLabel vec_delta_any phi_vec_cons_only;
rule vec_cons_any_signature ↪
  signature_cons
    vecLabel
    vec_delta_any
    consLabel
    constructor_list_nil
    top
    (signature_nil vecLabel vec_delta_any);

// Example signatures:
// Vec      = {Vec _ _     | empty, cons}
// NonEmpty = {Vec _ (S _) | cons}
symbol Vec : signature vecLabel vec_delta_any phi_vec_all;
rule Vec ↪ vec_signature;

symbol NonEmpty : signature vecLabel vec_delta_nonempty phi_vec_cons_only;
rule NonEmpty ↪ vec_nonempty_signature;

// Pattern requirement checks for NonEmpty signature.
symbol test_nonempty_pattern_accepts_cons : Bool;
rule test_nonempty_pattern_accepts_cons ↪
  constructor_in_signature_b
    vecLabel
    vec_delta_nonempty
    phi_vec_cons_only
    consLabel;

symbol test_nonempty_pattern_rejects_empty : Bool;
rule test_nonempty_pattern_rejects_empty ↪
  constructor_in_signature_b
    vecLabel
    vec_delta_nonempty
    phi_vec_cons_only
    emptyLabel;

assert ⊢ test_nonempty_pattern_accepts_cons ≡ true;
assert ⊢ test_nonempty_pattern_rejects_empty ≡ false;

// Signature-based API 1: NonEmpty -> nat (head-like extraction by match).
symbol nonempty_head_sig :
  signature_term vecLabel vec_delta_nonempty phi_vec_cons_only →
  nat;
rule nonempty_head_sig $m ↪
  signature_case
    vecLabel
    vec_delta_nonempty
    phi_vec_cons_only
    natLabel
    $m
    (case_branches_cons
      consLabel
      constructor_list_nil
      natLabel
      (λ (k : constructor_type consLabel), (+1 _0))
      (case_branches_nil natLabel));

// Signature-based API 2: Vector -> NonEmpty (append-style result).
symbol nonempty_term_code : Set;
rule (lift nonempty_term_code) ↪
  signature_term vecLabel vec_delta_nonempty phi_vec_cons_only;

symbol append_vec_to_nonempty :
  signature_term vecLabel vec_delta_any phi_vec_all →
  signature_term vecLabel vec_delta_nonempty phi_vec_cons_only;
rule append_vec_to_nonempty $m ↪
  signature_case
    vecLabel
    vec_delta_any
    phi_vec_all
    nonempty_term_code
    $m
    (case_branches_cons
      emptyLabel
      (constructor_list_cons consLabel constructor_list_nil)
      nonempty_term_code
      (λ (k : constructor_type emptyLabel),
        signature_intro
          vecLabel
          vec_delta_nonempty
          phi_vec_cons_only
          consLabel
          top
          cons)
      (case_branches_cons
        consLabel
        constructor_list_nil
        nonempty_term_code
        (λ (k : constructor_type consLabel),
          signature_intro
            vecLabel
            vec_delta_nonempty
            phi_vec_cons_only
            consLabel
            top
            cons)
        (case_branches_nil nonempty_term_code)));

// Examples: append-to-NonEmpty then extract with NonEmpty matcher.
symbol sample_vec_term_empty : signature_term vecLabel vec_delta_any phi_vec_all;
rule sample_vec_term_empty ↪
  signature_intro
    vecLabel
    vec_delta_any
    phi_vec_all
    emptyLabel
    top
    empty;

symbol sample_vec_term_cons : signature_term vecLabel vec_delta_any phi_vec_all;
rule sample_vec_term_cons ↪
  signature_intro
    vecLabel
    vec_delta_any
    phi_vec_all
    consLabel
    top
    cons;

symbol test_nonempty_head_after_append_empty : nat;
rule test_nonempty_head_after_append_empty ↪
  nonempty_head_sig (append_vec_to_nonempty sample_vec_term_empty);
assert ⊢ test_nonempty_head_after_append_empty ≡ (+1 _0);

symbol test_nonempty_head_after_append_cons : nat;
rule test_nonempty_head_after_append_cons ↪
  nonempty_head_sig (append_vec_to_nonempty sample_vec_term_cons);
assert ⊢ test_nonempty_head_after_append_cons ≡ (+1 _0);

print nat;
