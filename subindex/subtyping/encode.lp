
constant symbol Prop : TYPE;
builtin "Prop" ≔ Prop;

constant symbol Set : TYPE;
builtin "Set" ≔ Set;

injective symbol π : Prop → TYPE;
builtin "P" ≔ π;

symbol Data : Set;
symbol lift : Set → TYPE;

constant symbol ⊤ : Prop;
constant symbol top : π ⊤;

constant symbol ∧ : Prop → Prop → Prop; notation ∧ infix right 10;
constant symbol ∧ᵢ p q : π p → π q → π(p ∧ q);

inductive nat : TYPE ≔
| _0 : nat
| +1 : nat → nat; notation +1 postfix 100;

inductive Bool : TYPE ≔
| true : Bool
| false : Bool;

print nat; 

symbol notb : Bool → Bool;
rule notb true ↪ false;
rule notb false ↪ true;

// Bridge from booleans to propositions (used to enforce boolean side-conditions in types).
symbol is_true : Bool → Prop;
rule is_true true ↪ ⊤;

inductive labelSet : TYPE ≔
| natSetLabel : labelSet
| vecSetLabel : labelSet;

symbol natLabel : Set;
symbol vecLabel : Set;
symbol label : labelSet → Set;
rule natLabel ↪ label natSetLabel;
rule vecLabel ↪ label vecSetLabel;
rule (lift (label natSetLabel)) ↪ nat;

 inductive vec: Set → lift natLabel → TYPE ≔
| empty : Π (s : Set), vec s _0 // \Box
| cons : Π (s : Set) (n : nat), vec s n → lift s → vec s (+1 n); // ::

// vecLabel is just a tag for the family; no extra lift rule needed.
print vec;

// Codes for the vector family (so constructors can target a Set code).
symbol vecCode : Set → lift natLabel → Set;
rule (lift (vecCode $s $n)) ↪ vec $s $n;

constant symbol indexes_type : TYPE;
constant symbol index_null : indexes_type;
constant symbol index_dependent_lift_succ : Set → indexes_type → indexes_type;
constant symbol index_poly_succ : indexes_type → indexes_type;
constant symbol index_lift_succ : Set → indexes_type → indexes_type;
constant symbol index_lift_dep_succ : Π (k : Set), (lift k → indexes_type) → indexes_type;
constant symbol index_poly_dep_succ : (Set → indexes_type) → indexes_type;

symbol andb : Bool → Bool → Bool;
rule andb false $b ↪ false;
rule andb true $b ↪ $b;

symbol orb : Bool → Bool → Bool;
rule orb true $b ↪ true;
rule orb false $b ↪ $b;

symbol nat_eqb : nat → nat → Bool;
rule nat_eqb _0 _0 ↪ true;
rule nat_eqb _0 (+1 $n) ↪ false;
rule nat_eqb (+1 $n) _0 ↪ false;
rule nat_eqb (+1 $n1) (+1 $n2) ↪ nat_eqb $n1 $n2;

symbol label_set_eqb : labelSet → labelSet → Bool;
rule label_set_eqb natSetLabel natSetLabel ↪ true;
rule label_set_eqb natSetLabel vecSetLabel ↪ false;
rule label_set_eqb vecSetLabel natSetLabel ↪ false;
rule label_set_eqb vecSetLabel vecSetLabel ↪ true;

symbol datatype : Π (C : indexes_type) (s : Set), Set;
rule (lift (datatype (index_null) $s)) ↪ Set;
rule (lift (datatype (index_dependent_lift_succ $k $r) $s)) ↪ Π (a : lift $k), lift (datatype $r $s);
rule (lift (datatype (index_poly_succ $r) $s)) ↪ Π (a : Set), lift (datatype $r $s);
rule (lift (datatype (index_lift_succ $k $r) $s)) ↪ Π (a : lift $k), lift (datatype $r $s);
rule (lift (datatype (index_lift_dep_succ $k $r) $s)) ↪ Π (a : lift $k), lift (datatype ($r a) $s);
rule (lift (datatype (index_poly_dep_succ $r) $s)) ↪ Π (a : Set), lift (datatype ($r a) $s);

// Structural equality on Set codes used by constructor equality.
symbol set_eqb : Set → Set → Bool;

symbol lift_payload_eqb : Π (s1 : Set), lift s1 → Π (s2 : Set), lift s2 → Bool;
symbol lift_payload_eqb_nat_gate : Π (is_nat : Bool) (s1 : Set), lift s1 → Π (s2 : Set), lift s2 → Bool;
rule lift_payload_eqb $s1 $x1 $s2 $x2 ↪
  lift_payload_eqb_nat_gate (andb (set_eqb $s1 natLabel) (set_eqb $s2 natLabel)) $s1 $x1 $s2 $x2;
rule lift_payload_eqb_nat_gate false $s1 $x1 $s2 $x2 ↪ true;
rule lift_payload_eqb_nat_gate true (label natSetLabel) $n1 (label natSetLabel) $n2 ↪ nat_eqb $n1 $n2;

symbol indexes_type_eqb : indexes_type → indexes_type → Bool;
symbol indexes_tag : indexes_type → nat;
symbol indexes_type_eqb_gate : Bool → indexes_type → indexes_type → Bool;

rule set_eqb (label $l1) (label $l2) ↪ label_set_eqb $l1 $l2;
rule set_eqb (vecCode $s1 $n1) (vecCode $s2 $n2) ↪
  andb (set_eqb $s1 $s2) (nat_eqb $n1 $n2);
rule set_eqb (vecCode $s $n) (label $l) ↪ label_set_eqb vecSetLabel $l;
rule set_eqb (label $l) (vecCode $s $n) ↪ label_set_eqb $l vecSetLabel;
rule set_eqb (datatype $C $s) (label $l) ↪ false;
rule set_eqb (label $l) (datatype $C $s) ↪ false;
rule set_eqb (vecCode $s $n) (datatype $C $t) ↪ false;
rule set_eqb (datatype $C $t) (vecCode $s $n) ↪ false;
rule set_eqb (datatype $C1 $s1) (datatype $C2 $s2) ↪
  andb (indexes_type_eqb $C1 $C2) (set_eqb $s1 $s2);

rule indexes_tag index_null ↪ _0;
rule indexes_tag (index_dependent_lift_succ $s $r) ↪ (+1 _0);
rule indexes_tag (index_poly_succ $r) ↪ (+1 (+1 _0));
rule indexes_tag (index_lift_succ $k $r) ↪ (+1 (+1 (+1 _0)));
rule indexes_tag (index_lift_dep_succ $k $r) ↪ (+1 (+1 (+1 (+1 _0))));
rule indexes_tag (index_poly_dep_succ $r) ↪ (+1 (+1 (+1 (+1 (+1 _0)))));

rule indexes_type_eqb $C1 $C2 ↪
  indexes_type_eqb_gate (nat_eqb (indexes_tag $C1) (indexes_tag $C2)) $C1 $C2;
rule indexes_type_eqb_gate false $C1 $C2 ↪ false;
rule indexes_type_eqb_gate true index_null index_null ↪ true;
rule indexes_type_eqb_gate true (index_dependent_lift_succ $s1 $r1) (index_dependent_lift_succ $s2 $r2) ↪
  andb (set_eqb $s1 $s2) (indexes_type_eqb $r1 $r2);
rule indexes_type_eqb_gate true (index_poly_succ $r1) (index_poly_succ $r2) ↪
  indexes_type_eqb $r1 $r2;
rule indexes_type_eqb_gate true (index_lift_succ $k1 $r1) (index_lift_succ $k2 $r2) ↪
  andb (set_eqb $k1 $k2) (indexes_type_eqb $r1 $r2);
rule indexes_type_eqb_gate true (index_lift_dep_succ $k1 $r1) (index_lift_dep_succ $k2 $r2) ↪
  set_eqb $k1 $k2;
rule indexes_type_eqb_gate true (index_poly_dep_succ $r1) (index_poly_dep_succ $r2) ↪
  true;

symbol witness : indexes_type → TYPE;
symbol witness_constructor_dependent_lift : Π (r : indexes_type) (s : Set), witness r → lift s → witness (index_dependent_lift_succ s r);
// Polymorphic witness: provides a witness for each Set parameter.
symbol witness_constructor_poly : Π (r : indexes_type), (Set → witness r) → witness (index_poly_succ r);
symbol witness_constructor_lift : Π (r : indexes_type) (k : Set), witness r → witness (index_lift_succ k r);
symbol witness_constructor_lift_dep : Π (k : Set) (r : lift k → indexes_type), (Π (a : lift k), witness (r a)) → witness (index_lift_dep_succ k r);
symbol witness_constructor_poly_dep : Π (r : Set → indexes_type), (Π (a : Set), witness (r a)) → witness (index_poly_dep_succ r);
symbol witness_constructor_null : witness index_null;

symbol witness_eqb : Π (C1 : indexes_type), witness C1 → Π (C2 : indexes_type), witness C2 → Bool;
symbol witness_eqb_match : Π (ok : Bool) (C1 : indexes_type), witness C1 → Π (C2 : indexes_type), witness C2 → Bool;
rule witness_eqb $C1 $w1 $C2 $w2 ↪
  witness_eqb_match (indexes_type_eqb $C1 $C2) $C1 $w1 $C2 $w2;
rule witness_eqb_match false $C1 $w1 $C2 $w2 ↪ false;
rule witness_eqb_match true index_null witness_constructor_null index_null witness_constructor_null ↪ true;
// Compare witness structure; lifted payloads are compared when their label is natLabel.
rule witness_eqb_match true (index_dependent_lift_succ $s1 $r1) (witness_constructor_dependent_lift $r1 $s1 $w1 $a1)
                        (index_dependent_lift_succ $s2 $r2) (witness_constructor_dependent_lift $r2 $s2 $w2 $a2) ↪
  andb (set_eqb $s1 $s2)
    (andb (lift_payload_eqb $s1 $a1 $s2 $a2)
      (witness_eqb $r1 $w1 $r2 $w2));
rule witness_eqb_match true (index_poly_succ $r1) (witness_constructor_poly $r1 $wf1)
                        (index_poly_succ $r2) (witness_constructor_poly $r2 $wf2) ↪
  indexes_type_eqb $r1 $r2;
rule witness_eqb_match true (index_lift_succ $k1 $r1) (witness_constructor_lift $r1 $k1 $w1)
                        (index_lift_succ $k2 $r2) (witness_constructor_lift $r2 $k2 $w2) ↪
  andb (set_eqb $k1 $k2) (witness_eqb $r1 $w1 $r2 $w2);
rule witness_eqb_match true (index_lift_dep_succ $k1 $r1) (witness_constructor_lift_dep $k1 $r1 $wf1)
                        (index_lift_dep_succ $k2 $r2) (witness_constructor_lift_dep $k2 $r2 $wf2) ↪
  set_eqb $k1 $k2;
rule witness_eqb_match true (index_poly_dep_succ $r1) (witness_constructor_poly_dep $r1 $wf1)
                        (index_poly_dep_succ $r2) (witness_constructor_poly_dep $r2 $wf2) ↪
  true;

symbol datatype_eqb : Set → Set → Bool;
rule datatype_eqb (datatype $C1 $s1) (datatype $C2 $s2) ↪
  andb (indexes_type_eqb $C1 $C2) (set_eqb $s1 $s2);

symbol constructor_type_generic : Π (C : indexes_type) (s : Set) (h: witness C) (datatype: Set), lift datatype → TYPE;
rule (constructor_type_generic (index_null) $s (witness_constructor_null) (datatype index_null $s)) $M ↪ lift $M;
rule (constructor_type_generic (index_poly_succ $r) $s (witness_constructor_poly $r $wf)) (datatype (index_poly_succ $r) $s) $M ↪
    Π (a : Set), constructor_type_generic $r $s ($wf a) (datatype $r $s) ($M a);
rule (constructor_type_generic (index_dependent_lift_succ $a $r) $s (witness_constructor_dependent_lift $r $a $w' $a')) (datatype (index_dependent_lift_succ $a $r) $s) $M ↪
    constructor_type_generic $r $s $w' (datatype $r $s) ($M $a');
rule (constructor_type_generic (index_lift_succ $k $r) $s (witness_constructor_lift $r $k $w')) (datatype (index_lift_succ $k $r) $s) $M ↪
    Π (a : lift $k), constructor_type_generic $r $s $w' (datatype $r $s) ($M a);
rule (constructor_type_generic (index_lift_dep_succ $k $r) $s (witness_constructor_lift_dep $k $r $wf)) (datatype (index_lift_dep_succ $k $r) $s) $M ↪
    Π (a : lift $k), constructor_type_generic ($r a) $s ($wf a) (datatype ($r a) $s) ($M a);
rule (constructor_type_generic (index_poly_dep_succ $r) $s (witness_constructor_poly_dep $r $wf)) (datatype (index_poly_dep_succ $r) $s) $M ↪
    Π (a : Set), constructor_type_generic ($r a) $s ($wf a) (datatype ($r a) $s) ($M a);

symbol constructor_type_eqb :
  Π (C1 : indexes_type) (s1 : Set) (h1 : witness C1) (datatype1 : Set), lift datatype1 →
  Π (C2 : indexes_type) (s2 : Set) (h2 : witness C2) (datatype2 : Set), lift datatype2 →
  Bool;
rule constructor_type_eqb $C1 $s1 $h1 $datatype1 $M1 $C2 $s2 $h2 $datatype2 $M2 ↪
  andb (indexes_type_eqb $C1 $C2)
    (andb (set_eqb $s1 $s2)
      (andb (witness_eqb $C1 $h1 $C2 $h2)
        (datatype_eqb $datatype1 $datatype2)));

symbol constructor_type_diffb :
  Π (C1 : indexes_type) (s1 : Set) (h1 : witness C1) (datatype1 : Set), lift datatype1 →
  Π (C2 : indexes_type) (s2 : Set) (h2 : witness C2) (datatype2 : Set), lift datatype2 →
  Bool;
rule constructor_type_diffb $C1 $s1 $h1 $datatype1 $M1 $C2 $s2 $h2 $datatype2 $M2 ↪
  notb (constructor_type_eqb $C1 $s1 $h1 $datatype1 $M1 $C2 $s2 $h2 $datatype2 $M2);


//Π [s : Set], vec s _0;
constant symbol mk_constructor : Π [s : Set], vec s _0;

// Constructor labels and lookup table to constructor values.
inductive labelConstructor : TYPE ≔
| emptyLabel : labelConstructor
| consLabel : labelConstructor;

symbol constructor_type : labelConstructor → TYPE;
rule constructor_type emptyLabel ↪
  (constructor_type_generic
    (index_poly_succ (index_dependent_lift_succ natLabel index_null))
    vecLabel
    (witness_constructor_poly
      (index_dependent_lift_succ natLabel index_null)
      (λ (A : Set),
        witness_constructor_dependent_lift
          index_null
          natLabel
          witness_constructor_null
          _0))
    (datatype (index_poly_succ (index_dependent_lift_succ natLabel index_null)) vecLabel)
    vecCode);
rule constructor_type consLabel ↪
  (constructor_type_generic
    (index_poly_dep_succ
      (λ (A : Set),
        index_lift_dep_succ natLabel
          (λ (n : nat),
            index_lift_succ (vecCode A n)
              (index_lift_succ A index_null))))
    vecLabel
    (witness_constructor_poly_dep
      (λ (A : Set),
        index_lift_dep_succ natLabel
          (λ (n : nat),
            index_lift_succ (vecCode A n)
              (index_lift_succ A index_null)))
      (λ (A : Set),
        witness_constructor_lift_dep
          natLabel
          (λ (n : nat),
            index_lift_succ (vecCode A n)
              (index_lift_succ A index_null))
          (λ (n : nat),
            witness_constructor_lift
              (index_lift_succ A index_null)
              (vecCode A n)
              (witness_constructor_lift
                index_null
                A
                witness_constructor_null))))
    (datatype
      (index_poly_dep_succ
        (λ (A : Set),
          index_lift_dep_succ natLabel
            (λ (n : nat),
              index_lift_succ (vecCode A n)
                (index_lift_succ A index_null))))
      vecLabel)
    (λ (A : Set) (n : nat) (v : vec A n) (x : lift A), vecCode A (+1 n)));

// Rule 1 abstraction (AGAINST side-condition): generic constructor patterns for indices.
// This avoids hard-coding one AGAINST relation per inductive datatype.
symbol index_term_list : TYPE;

inductive index_term : TYPE ≔
| index_term_var : index_term
| index_term_ctor : nat → index_term_list → index_term;

symbol index_term_nil : index_term_list;
symbol index_term_cons : index_term → index_term_list → index_term_list;

inductive index_pattern_atom : TYPE ≔
| index_pat_var : index_pattern_atom
| index_pat_term : index_term → index_pattern_atom;

inductive index_pattern_list : TYPE ≔
| index_pattern_nil : index_pattern_list
| index_pattern_cons : index_pattern_atom → index_pattern_list → index_pattern_list;

symbol constructor_pair : TYPE;
symbol index_term_subst_nat : nat → index_term → nat;
symbol mk_constructor_pair :
  Π (l : labelConstructor),
  constructor_type l →
  nat →
  Set →
  indexes_type →
  index_pattern_list →
  (Set → nat → index_pattern_list → Set) →
  constructor_pair;

// Constructor-result interpreter metadata for the vec family.
symbol vec_result_from_pattern : Set → nat → index_pattern_list → Set;
rule vec_result_from_pattern
  $A
  $n
  (index_pattern_cons index_pat_var (index_pattern_cons (index_pat_term $t) index_pattern_nil)) ↪
  vecCode $A (index_term_subst_nat $n $t);

symbol constructor : Π (l : labelConstructor), constructor_pair;
rule constructor emptyLabel ↪
  mk_constructor_pair
    emptyLabel
    empty
    _0
    vecLabel
    (index_poly_succ (index_dependent_lift_succ natLabel index_null))
    (index_pattern_cons index_pat_var
      (index_pattern_cons
        (index_pat_term (index_term_ctor _0 index_term_nil))
        index_pattern_nil))
    vec_result_from_pattern;
rule constructor consLabel ↪
  mk_constructor_pair
    consLabel
    cons
    (+1 _0)
    vecLabel
    (index_poly_succ (index_dependent_lift_succ natLabel index_null))
    (index_pattern_cons index_pat_var
      (index_pattern_cons
        (index_pat_term
          (index_term_ctor
            (+1 _0)
            (index_term_cons index_term_var index_term_nil)))
        index_pattern_nil))
    vec_result_from_pattern;

// Decidable equality on constructor labels (used internally by constructor_eqb).
symbol constructor_label_eqb : labelConstructor → labelConstructor → Bool;
rule constructor_label_eqb emptyLabel emptyLabel ↪ true;
rule constructor_label_eqb emptyLabel consLabel ↪ false;
rule constructor_label_eqb consLabel emptyLabel ↪ false;
rule constructor_label_eqb consLabel consLabel ↪ true;

// Decidable equality on constructor values.
symbol constructor_eqb : constructor_pair → constructor_pair → Bool;
rule constructor_eqb (mk_constructor_pair $l1 $c1 $n1 $T1 $I1 $p1 $interp1) (mk_constructor_pair $l2 $c2 $n2 $T2 $I2 $p2 $interp2) ↪
  andb (constructor_label_eqb $l1 $l2)
    (andb (nat_eqb $n1 $n2)
      (andb (set_eqb $T1 $T2) (indexes_type_eqb $I1 $I2)));

// Constructor-list signatures.
inductive constructor_list : TYPE ≔
| constructor_list_nil : constructor_list
| constructor_list_cons : labelConstructor → constructor_list → constructor_list;

// Directional AGAINST for generic constructor terms.
symbol against_term : index_term → index_term → Bool;
symbol against_term_list : index_term_list → index_term_list → Bool;
rule against_term index_term_var $q ↪ true;
rule against_term (index_term_ctor $c $ps) index_term_var ↪ false;
rule against_term (index_term_ctor $c1 $ps1) (index_term_ctor $c2 $ps2) ↪
  andb (nat_eqb $c1 $c2) (against_term_list $ps1 $ps2);
rule against_term_list index_term_nil index_term_nil ↪ true;
rule against_term_list index_term_nil (index_term_cons $q $qs) ↪ false;
rule against_term_list (index_term_cons $p $ps) index_term_nil ↪ false;
rule against_term_list (index_term_cons $p $ps) (index_term_cons $q $qs) ↪
  andb (against_term $p $q) (against_term_list $ps $qs);

// Directional AGAINST for one index atom.
symbol against_atom : index_pattern_atom → index_pattern_atom → Bool;
rule against_atom index_pat_var $q ↪ true;
rule against_atom (index_pat_term $p) index_pat_var ↪ false;
rule against_atom (index_pat_term $p) (index_pat_term $q) ↪ against_term $p $q;

// Directional AGAINST for index tuples (Δ* against constructor result indices).
symbol against_pattern_list : index_pattern_list → index_pattern_list → Bool;
rule against_pattern_list index_pattern_nil index_pattern_nil ↪ true;
rule against_pattern_list index_pattern_nil (index_pattern_cons $q $qs) ↪ false;
rule against_pattern_list (index_pattern_cons $p $ps) index_pattern_nil ↪ false;
rule against_pattern_list (index_pattern_cons $p $ps) (index_pattern_cons $q $qs) ↪
  andb (against_atom $p $q) (against_pattern_list $ps $qs);

// Equality on index patterns (used by Rule 4 abstraction).
symbol pattern_list_eqb : index_pattern_list → index_pattern_list → Bool;
rule pattern_list_eqb $p1 $p2 ↪
  andb (against_pattern_list $p1 $p2) (against_pattern_list $p2 $p1);

// Metadata extractors from the constructor database.
symbol constructor_target : constructor_pair → Set;
rule constructor_target (mk_constructor_pair $l $c $n $T $I $p $interp) ↪ $T;

symbol constructor_indexes : constructor_pair → indexes_type;
rule constructor_indexes (mk_constructor_pair $l $c $n $T $I $p $interp) ↪ $I;

symbol constructor_result_pattern : constructor_pair → index_pattern_list;
rule constructor_result_pattern (mk_constructor_pair $l $c $n $T $I $p $interp) ↪ $p;

symbol constructor_result_interp : constructor_pair → Set → nat → index_pattern_list → Set;
rule constructor_result_interp (mk_constructor_pair $l $c $n $T $I $p $interp) $A $m $g ↪
  $interp $A $m $g;

// Constructor index used to align constructor order with eliminator branch order.
symbol constructor_rank : constructor_pair → nat;
rule constructor_rank (mk_constructor_pair $l $c $n $T $I $p $interp) ↪ $n;

symbol constructor_case_index : labelConstructor → nat;
rule constructor_case_index $c ↪ constructor_rank (constructor $c);

// AGAINST side-condition: family matches and index pattern is general enough.
symbol againstb : Set → index_pattern_list → constructor_pair → Bool;
rule againstb $T $g $k ↪
  andb (set_eqb (constructor_target $k) $T)
    (against_pattern_list $g (constructor_result_pattern $k));

// Automatic interpretation: lift (constructor, [index_pattern]) ↪ constructor applied to those indices.
// For now we interpret nat-like index terms encoded with constructor ids 0 and 1 (succ).
rule index_term_subst_nat $x index_term_var ↪ $x;
rule index_term_subst_nat $x (index_term_ctor _0 index_term_nil) ↪ _0;
rule index_term_subst_nat $x (index_term_ctor (+1 _0) (index_term_cons $t index_term_nil)) ↪
  +1 (index_term_subst_nat $x $t);

symbol constructor_result_code_from_pattern :
  constructor_pair → Set → nat → index_pattern_list → Set;
rule constructor_result_code_from_pattern $k $A $n $g ↪
  constructor_result_interp $k $A $n $g;

symbol lift_constructor_from_pattern :
  constructor_pair → Set → nat → index_pattern_list → TYPE;
rule lift_constructor_from_pattern $k $A $n $g ↪
  lift (constructor_result_code_from_pattern $k $A $n $g);

// Δ encoding (paper-style): index shape + index witness-pattern.
symbol delta_tel : TYPE;
symbol mk_delta_tel : indexes_type → index_pattern_list → delta_tel;

symbol delta_indexes : delta_tel → indexes_type;
rule delta_indexes (mk_delta_tel $I $w) ↪ $I;

symbol delta_witness : delta_tel → index_pattern_list;
rule delta_witness (mk_delta_tel $I $w) ↪ $w;

symbol delta_eqb : delta_tel → delta_tel → Bool;
rule delta_eqb $g1 $g2 ↪
  andb
    (indexes_type_eqb (delta_indexes $g1) (delta_indexes $g2))
    (pattern_list_eqb (delta_witness $g1) (delta_witness $g2));

// Normalization for T Δ using constructor metadata + Δ witness-pattern.
symbol normalize_tdelta_code : constructor_pair → Set → nat → delta_tel → Set;
rule normalize_tdelta_code $k $A $n $g ↪
  constructor_result_code_from_pattern $k $A $n (delta_witness $g);

symbol normalize_tdelta : constructor_pair → Set → nat → delta_tel → TYPE;
rule normalize_tdelta $k $A $n $g ↪
  lift (normalize_tdelta_code $k $A $n $g);

// Rule 3 abstraction (constructor admissibility): membership in Φ.
symbol constructor_mem_b : labelConstructor → constructor_list → Bool;
rule constructor_mem_b $c constructor_list_nil ↪ false;
rule constructor_mem_b $c (constructor_list_cons $c' $cs) ↪
  orb (constructor_label_eqb $c $c') (constructor_mem_b $c $cs);

// Rule 4 abstraction helper: subset test on constructor lists.
symbol constructor_list_subsetb : constructor_list → constructor_list → Bool;
rule constructor_list_subsetb constructor_list_nil $ys ↪ true;
rule constructor_list_subsetb (constructor_list_cons $x $xs) $ys ↪
  andb (constructor_mem_b $x $ys) (constructor_list_subsetb $xs $ys);

// AGAINST with explicit Δ context.
symbol against_delta_b : Set → delta_tel → constructor_pair → Bool;
rule against_delta_b $T $g $k ↪
  andb (indexes_type_eqb (delta_indexes $g) (constructor_indexes $k))
    (againstb $T (delta_witness $g) $k);

// Rule 3 abstraction: constructor can inhabit {T Δ* :: Φ} when it is listed in Φ and passes AGAINST.
symbol constructor_in_signature_b : Set → delta_tel → constructor_list → labelConstructor → Bool;
rule constructor_in_signature_b $T $g $phi $c ↪
  andb (constructor_mem_b $c $phi) (against_delta_b $T $g (constructor $c));

// Rule 1 abstraction helper: full well-formedness check for (TΔ, Φ).
symbol signature_wfb : Set → delta_tel → constructor_list → Bool;
rule signature_wfb $T $g constructor_list_nil ↪ true;
rule signature_wfb $T $g (constructor_list_cons $c $cs) ↪
  andb (against_delta_b $T $g (constructor $c)) (signature_wfb $T $g $cs);

// Rule 1 abstraction (formation): signature receives both T and Δ.
inductive signature : Set → delta_tel → constructor_list → TYPE ≔
| signature_nil : Π (T : Set) (g : delta_tel), signature T g constructor_list_nil
| signature_cons :
    Π (T : Set) (g : delta_tel) (c : labelConstructor) (cs : constructor_list),
    π (is_true (against_delta_b T g (constructor c))) →
    signature T g cs →
    signature T g (constructor_list_cons c cs);

// Rule 2 abstraction: {TΔ :: Φ} ≤ TΔ (forget signature refinement).
symbol tdelta_base_eqb : Set → delta_tel → Set → delta_tel → Bool;
rule tdelta_base_eqb $T1 $g1 $T2 $g2 ↪
  andb (set_eqb $T1 $T2) (delta_eqb $g1 $g2);

symbol signature_to_base_subtypeb : Set → delta_tel → constructor_list → Set → delta_tel → Bool;
rule signature_to_base_subtypeb $T $g $phi $U $g' ↪
  andb (signature_wfb $T $g $phi)
    (tdelta_base_eqb $T $g $U $g');

// Rule 4 abstraction: {T'Δ :: Φ'} ≤ {TΔ :: Φ} when base types match and Φ' ⊆ Φ.
symbol signature_subtypeb : Set → delta_tel → constructor_list → Set → delta_tel → constructor_list → Bool;
rule signature_subtypeb $T1 $g1 $phi1 $T2 $g2 $phi2 ↪
  andb (signature_wfb $T1 $g1 $phi1)
    (andb (signature_wfb $T2 $g2 $phi2)
      (andb (tdelta_base_eqb $T1 $g1 $T2 $g2)
        (constructor_list_subsetb $phi1 $phi2)));

// Rule 5 judgmental encoding:
// if F is not a registered constructor and inferred A is a subtype of the target signature, then F Δ checks against it.
symbol rule5_checkb :
  Bool →
  Set → delta_tel → constructor_list →
  Set → delta_tel → constructor_list →
  Bool;
rule rule5_checkb $is_constructor $T $g $phi_target $A $gA $phiA ↪
  andb (notb $is_constructor)
    (signature_subtypeb $A $gA $phiA $T $g $phi_target);

symbol rule5_jdg :
  Bool →
  Set → delta_tel → constructor_list →
  Set → delta_tel → constructor_list →
  Prop;
rule rule5_jdg $is_constructor $T $g $phi_target $A $gA $phiA ↪
  is_true (rule5_checkb $is_constructor $T $g $phi_target $A $gA $phiA);

// Rule 6 judgmental encoding (arrow subtyping, non-dependent approximation):
// (x:A)->B ≤ (x:A')->B' when A' ≤ A (contravariant domain) and B ≤ B' (covariant codomain).
symbol pi_signature : TYPE;
symbol mk_pi_signature :
  Set → delta_tel → constructor_list →
  Set → delta_tel → constructor_list →
  pi_signature;

symbol pi_signature_subtypeb : pi_signature → pi_signature → Bool;
rule pi_signature_subtypeb
  (mk_pi_signature $A $gA $phiA $B $gB $phiB)
  (mk_pi_signature $A' $gA' $phiA' $B' $gB' $phiB') ↪
  andb
    (signature_subtypeb $A' $gA' $phiA' $A $gA $phiA)
    (signature_subtypeb $B $gB $phiB $B' $gB' $phiB');

symbol pi_signature_subtype : pi_signature → pi_signature → Prop;
rule pi_signature_subtype $p1 $p2 ↪
  is_true (pi_signature_subtypeb $p1 $p2);

// Rule 7 abstraction (case analysis over signatures).
// A scrutinee carries the constructor label and its constructor payload.
inductive signature_term : Set → delta_tel → constructor_list → TYPE ≔
| signature_intro :
    Π (T : Set) (g : delta_tel) (phi : constructor_list) (c : labelConstructor),
    π (is_true (constructor_in_signature_b T g phi c)) →
    constructor_type c →
    signature_term T g phi;

// One branch per constructor in Φ, with branch type guided by constructor metadata.
inductive case_branches : constructor_list → Set → TYPE ≔
| case_branches_nil :
    Π (Q : Set),
    case_branches constructor_list_nil Q
| case_branches_cons :
    Π (c : labelConstructor) (cs : constructor_list) (Q : Set),
    (constructor_type c → lift Q) →
    case_branches cs Q →
    case_branches (constructor_list_cons c cs) Q;

// Dispatch for each constructor label; recursion follows constructor order in Φ.
symbol case_dispatch_empty :
  Π (phi : constructor_list) (Q : Set),
  constructor_type emptyLabel →
  case_branches phi Q →
  lift Q;

rule case_dispatch_empty
  (constructor_list_cons emptyLabel $cs)
  $Q
  $m
  (case_branches_cons emptyLabel $cs $Q $n $ns) ↪
  $n $m;

rule case_dispatch_empty
  (constructor_list_cons consLabel $cs)
  $Q
  $m
  (case_branches_cons consLabel $cs $Q $n $ns) ↪
  case_dispatch_empty $cs $Q $m $ns;

symbol case_dispatch_cons :
  Π (phi : constructor_list) (Q : Set),
  constructor_type consLabel →
  case_branches phi Q →
  lift Q;

rule case_dispatch_cons
  (constructor_list_cons consLabel $cs)
  $Q
  $m
  (case_branches_cons consLabel $cs $Q $n $ns) ↪
  $n $m;

rule case_dispatch_cons
  (constructor_list_cons emptyLabel $cs)
  $Q
  $m
  (case_branches_cons emptyLabel $cs $Q $n $ns) ↪
  case_dispatch_cons $cs $Q $m $ns;

// Rule 7 checker/eliminator:
// if M : {T Δ* :: Φ} and we have one branch per constructor in Φ, then case M of ... : Q.
symbol signature_case :
  Π (T : Set) (g : delta_tel) (phi : constructor_list) (Q : Set),
  signature_term T g phi →
  case_branches phi Q →
  lift Q;

rule signature_case $T $g $phi $Q (signature_intro $T $g $phi emptyLabel $ok $m) $bs ↪
  case_dispatch_empty $phi $Q $m $bs;

rule signature_case $T $g $phi $Q (signature_intro $T $g $phi consLabel $ok $m) $bs ↪
  case_dispatch_cons $phi $Q $m $bs;

// Automatic coercion layer for signature subtyping.
// `cast_signature` is primitive (opaque): it changes only the signature refinement.
symbol cast_signature :
  Π (T : Set) (g : delta_tel) (phi1 : constructor_list) (phi2 : constructor_list),
  signature T g phi1 → signature T g phi2;

symbol coerce_signature :
  Π (T : Set) (g : delta_tel) (phi1 : constructor_list) (phi2 : constructor_list),
  signature T g phi1 → signature T g phi2;

symbol coerce_signature_gate :
  Bool →
  Π (T : Set) (g : delta_tel) (phi1 : constructor_list) (phi2 : constructor_list),
  signature T g phi1 → signature T g phi2;

rule coerce_signature $T $g $phi1 $phi2 $x ↪
  coerce_signature_gate
    (signature_subtypeb $T $g $phi1 $T $g $phi2)
    $T
    $g
    $phi1
    $phi2
    $x;

// Coercion proceeds only when the subtype checker validates the relation.
rule coerce_signature_gate true $T $g $phi1 $phi2 $x ↪
  cast_signature $T $g $phi1 $phi2 $x;

// LambdaPi coercion hook: whenever expected type is a signature with the same T,Δ
// and a broader constructor list, insert `coerce_signature` automatically.
coerce_rule coerce_signature
  (signature $T $g $phi1)
  (signature $T $g $phi2)
  $x ↪ coerce_signature $T $g $phi1 $phi2 $x;
