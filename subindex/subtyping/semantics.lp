constant symbol kind : TYPE;
constant symbol set1 : kind;
constant symbol nat : TYPE;
constant symbol Z : nat;
constant symbol S : nat → nat;
symbol typed : kind → TYPE;
rule (typed set1) ↪ kind;
constant symbol unit : TYPE;
constant symbol u : unit;
constant symbol nothing: kind;
constant symbol par :  Π (k : kind) (k' : kind), kind;
constant symbol mk_par : Π (k : kind) (k' : kind), typed (par k k');
constant symbol indexes_type : TYPE;
symbol static_symbol : indexes_type → kind;
symbol arrow : kind → kind → kind;
constant symbol sigma : Π (P : Π (k: kind), kind), TYPE;
constant symbol mk_sigma : Π (P : Π (k: kind), kind) (k : kind), typed (P k) → (sigma P);


constant symbol index_null : indexes_type;
constant symbol index_succ : kind → indexes_type → indexes_type;

rule (typed (static_symbol (index_null))) ↪ kind;
rule (typed (static_symbol (index_succ $k $res))) ↪ Π (n : (typed $k)), (typed (static_symbol $res));

symbol pi_nul : kind;
symbol pi_append : kind → kind → kind;

rule typed (pi_nul) ↪ (typed nothing);
rule typed (pi_append $k $a) ↪ Π (n : (typed $k)), (typed $a);

symbol Constructor : Π (p: kind) (k: kind), kind;
rule (typed (Constructor pi_nul $k)) ↪ (typed $k);
rule (typed (Constructor (pi_append  ) $k)) ↪ typed $p → typed $k;

constant symbol signature : nat → kind → kind;
constant symbol signature_bottom : Π (k : kind), typed (signature Z k);
constant symbol signature_cons
  : Π (n : nat) (r: kind) (k : kind), typed (Constructor r k) → typed (signature n k) → typed (signature (S n) k);

symbol Nat : typed (static_symbol index_null);

// Declare NatSig first
symbol NatSig : typed (signature (S (S Z)) Nat);

// Interpret a signature as a kind (type)
// This creates a distinct type for each signature, preserving signature information
// No reduction rules - interpret remains abstract to maintain distinctness
symbol interpret : Π (n : nat) (k : kind), typed (signature n k) → kind;

// Injection function: inject values from base types into signature types
// Takes a value of a base type and wraps it as being typed by the signature
// This is opaque - no reduction - maintaining type distinction
symbol inject : Π (n : nat) (k : kind) (sig : typed (signature (S n) k)) (base : kind),
  typed base → typed (interpret (S n) k sig);

// Helper: convert a constructor kind to the type of its case handler
// For pi_nul: the case takes no arguments, just returns typed Q
// For pi_append: the case takes an argument and continues
symbol constructor_case_type : kind → kind → TYPE;
rule constructor_case_type (pi_nul $base) $Q ↪ typed $Q;
rule constructor_case_type (pi_append $arg $rest) $Q ↪
  Π (x : typed $arg), constructor_case_type $rest $Q;

// Helper: type representing cases for all constructors in a signature
// For signature_bottom (0 constructors): unit (no cases needed)
// For signature with 1 constructor: just the case type
// For signature with 2+ constructors: case for first, then Pi for rest
symbol match_cases : Π (n : nat) (k : kind), typed (signature n k) → kind → TYPE;
rule match_cases Z $k (signature_bottom $k) $Q ↪ unit;
rule match_cases (S Z) $k (signature_cons Z $k $p (signature_bottom $k)) $Q ↪
  constructor_case_type $k $Q;
rule match_cases (S (S $n)) $k (signature_cons (S $n) $r $k $p $rest) $Q ↪
  Π (case : constructor_case_type $k $Q), match_cases (S $n) $k $rest $Q;

// Match/case expression for interpreted signature values
// Takes a value of type typed (interpret n sig) and cases for each constructor
// Returns a result of type typed Q
symbol match_interpret :
  Π (n : nat) (k : kind) (sig : typed (signature n k)) (Q : kind),
  typed (interpret n k sig) →  // Value to match on
  match_cases n k sig Q →      // Cases for each constructor in sig
  typed Q;

// Base constructors for internal use only
symbol z_base : typed Nat;
symbol succ_base : typed Nat → typed Nat;  // Base successor for internal signatures

// Public constructors work on signature types
symbol z : typed (interpret (S (S Z)) Nat NatSig);
rule z ↪ inject (S Z) Nat NatSig Nat z_base;

symbol s : typed (interpret (S (S Z)) Nat NatSig) → typed Nat;  // Destructor for NatSig
symbol succ : typed (interpret (S (S Z)) Nat NatSig) → typed (interpret (S (S Z)) Nat NatSig);  // Successor for signature-typed Nat

assert ⊢ (typed Nat) ≡ typed (Constructor nothing Nat);

// Complete Nat signature with both z and s constructors
rule NatSig ↪ signature_cons (S Z)
                nothing
                Nat  // z constructor: no arguments, returns Nat
                z_base
                  (signature_bottom Nat);

// Subtyping relation: signature_bottom is a subtype of everything,
// and signature_cons inherits subtyping from its parent signature
symbol subtype : Π (n : nat) (m : nat), typed (signature n) → typed (signature m) → TYPE;

// signature_bottom < X for all X
rule subtype Z $m signature_bottom _ ↪ unit;

// (signature_cons $n $k $p $rest) < Y if $rest < Y
// The signature_cons is a subtype if its parent signature is a subtype
rule subtype (S $n) $m (signature_cons $n $k $p $rest) $sig ↪ subtype $n $m $rest $sig;

// Vector base type indexed by element type and length (polymorphic)
symbol Vector : typed (static_symbol (index_succ set1 (index_succ (interpret (S (S Z)) NatSig) index_null)));

// VectorSig - signature family parameterized by element type and length
symbol VectorSig : Π (A : typed set1) (n : typed (interpret (S (S Z)) NatSig)), typed (signature (S (S Z)));

// Base constructors for vectors (internal use)
symbol empty_base : Π (A : typed set1), typed (Vector A z);
symbol cons_base : Π (A : typed set1) (n : typed (interpret (S (S Z)) NatSig)), typed A → typed (interpret (S (S Z)) (VectorSig A n)) → typed (Vector A (succ n));

// Define VectorSig with empty and cons constructors
// Each VectorSig A n describes operations for vectors of element type A and length n
rule VectorSig $A $n ↪ signature_cons (S Z)
  (pi_nul (Vector $A z))  // empty constructor: returns Vector A z
  (empty_base $A)
  (signature_cons Z
    (pi_append $A (pi_append (interpret (S (S Z)) (VectorSig $A $n)) (pi_nul (Vector $A (succ $n)))))  // cons constructor
    (cons_base $A $n)
    signature_bottom);

// Public constructors work on signature types
symbol empty : Π (A : typed set1), typed (interpret (S (S Z)) (VectorSig A z));
rule empty $A ↪ inject (S Z) (VectorSig $A z) (Vector $A z) (empty_base $A);

symbol cons : Π (A : typed set1) (n : typed (interpret (S (S Z)) NatSig)), typed A → typed (interpret (S (S Z)) (VectorSig A n)) → typed (interpret (S (S Z)) (VectorSig A (succ n)));
rule cons $A $n $elem $vec ↪ inject (S Z) (VectorSig $A (succ $n)) (Vector $A (succ $n)) (cons_base $A $n $elem $vec);

// Helper function to create a signature with a single constructor
// Takes a constructor kind and the constructor itself
symbol mk_signature : Π (k : kind), typed k → typed (signature (S Z));
rule mk_signature $k $p ↪ signature_cons Z $k $p signature_bottom;

// NonEmptyVector signature - represents non-empty vectors
// Has only the cons constructor (no empty), so pattern matching requires only one case
symbol cons_vec_base : Π (A : typed set1) (n : typed (interpret (S (S Z)) NatSig)), typed A → typed (interpret (S (S Z)) (VectorSig A n)) → typed (Vector A (succ n));

symbol NonEmptyVector : Π (A : typed set1) (n : typed (interpret (S (S Z)) NatSig)), typed (signature (S Z));
rule NonEmptyVector $A $n ↪ mk_signature
  (pi_append $A (pi_append (interpret (S (S Z)) (VectorSig $A $n)) (pi_nul (Vector $A (succ $n)))))
  (cons_vec_base $A $n);

// Head function for non-empty vectors
// Demonstrates match_interpret with only ONE case (since NonEmptyVector has only cons)
symbol head : Π (A : typed set1) (n : typed (interpret (S (S Z)) NatSig)), typed (interpret (S Z) (NonEmptyVector A n)) → typed A;
rule head $A $n $v ↪
  match_interpret (S Z) (NonEmptyVector $A $n) $A $v
    (λ elem vec, elem);  // Single case: extract the head element

// Test: Create a non-empty vector and extract its head
// This demonstrates that match_interpret with NonEmptyVector requires only ONE case
// (compared to regular Vector which would need two cases: empty and cons)
symbol test_nev : typed (interpret (S Z) (NonEmptyVector Nat z));
rule test_nev ↪ inject Z (NonEmptyVector Nat z) (Vector Nat (succ z))
  (cons_vec_base Nat z z_base (empty Nat));

// Test: head extracts the first element
assert ⊢ head Nat z test_nev : typed Nat;

// Test: Create vectors with actual values injected
// Empty vector of Nat
symbol vec0 : typed (interpret (S (S Z)) (VectorSig Nat z));
rule vec0 ↪ empty Nat;

// Vector with one Nat element
symbol vec1 : typed (interpret (S (S Z)) (VectorSig Nat (succ z)));
rule vec1 ↪ cons Nat z z_base (empty Nat);

// Vector with two Nat elements
symbol vec2 : typed (interpret (S (S Z)) (VectorSig Nat (succ (succ z))));
rule vec2 ↪ cons Nat (succ z) (succ_base z_base) (cons Nat z z_base (empty Nat));

// Vector with three Nat elements
symbol vec3 : typed (interpret (S (S Z)) (VectorSig Nat (succ (succ (succ z)))));
rule vec3 ↪ cons Nat (succ (succ z)) (succ_base (succ_base z_base))
  (cons Nat (succ z) (succ_base z_base)
    (cons Nat z z_base (empty Nat)));

// Test: Verify the types
assert ⊢ vec0 : typed (interpret (S (S Z)) (VectorSig Nat z));
assert ⊢ vec1 : typed (interpret (S (S Z)) (VectorSig Nat (succ z)));
assert ⊢ vec2 : typed (interpret (S (S Z)) (VectorSig Nat (succ (succ z))));
assert ⊢ vec3 : typed (interpret (S (S Z)) (VectorSig Nat (succ (succ (succ z)))));

// Test: Create a non-empty vector with actual value and extract head
symbol vec1_nonempty : typed (interpret (S Z) (NonEmptyVector Nat z));
rule vec1_nonempty ↪ inject Z (NonEmptyVector Nat z) (Vector Nat (succ z))
  (cons_vec_base Nat z (succ_base (succ_base z_base)) (empty Nat));

// Extract the head - should return (succ_base (succ_base z_base))
assert ⊢ head Nat z vec1_nonempty : typed Nat;

// Demonstrate head function extracting values using match_interpret
// Create non-empty vector with value z_base (0)
symbol nev_0 : typed (interpret (S Z) (NonEmptyVector Nat z));
rule nev_0 ↪ inject Z (NonEmptyVector Nat z) (Vector Nat (succ z))
  (cons_vec_base Nat z z_base (empty Nat));

// Extract head - should give z_base
symbol extracted_0 : typed Nat;
rule extracted_0 ↪ head Nat z nev_0;

// Create non-empty vector with value (succ_base z_base) = 1
symbol nev_1 : typed (interpret (S Z) (NonEmptyVector Nat z));
rule nev_1 ↪ inject Z (NonEmptyVector Nat z) (Vector Nat (succ z))
  (cons_vec_base Nat z (succ_base z_base) (empty Nat));

// Extract head - should give (succ_base z_base)
symbol extracted_1 : typed Nat;
rule extracted_1 ↪ head Nat z nev_1;

// Create non-empty vector with value (succ_base (succ_base z_base)) = 2
symbol nev_2 : typed (interpret (S Z) (NonEmptyVector Nat z));
rule nev_2 ↪ inject Z (NonEmptyVector Nat z) (Vector Nat (succ z))
  (cons_vec_base Nat z (succ_base (succ_base z_base)) (empty Nat));

// Extract head - should give (succ_base (succ_base z_base))
symbol extracted_2 : typed Nat;
rule extracted_2 ↪ head Nat z nev_2;

// Verify the extracted values have correct type
assert ⊢ extracted_0 : typed Nat;
assert ⊢ extracted_1 : typed Nat;
assert ⊢ extracted_2 : typed Nat;

// Create a longer non-empty vector (length 2) and extract head
symbol nev_longer : typed (interpret (S Z) (NonEmptyVector Nat (succ z)));
rule nev_longer ↪ inject Z (NonEmptyVector Nat (succ z)) (Vector Nat (succ (succ z)))
  (cons_vec_base Nat (succ z) (succ_base (succ_base (succ_base z_base))) (cons Nat z (succ_base z_base) (empty Nat)));

// Extract head from longer vector - should give (succ_base (succ_base (succ_base z_base))) = 3
symbol extracted_from_longer : typed Nat;
rule extracted_from_longer ↪ head Nat (succ z) nev_longer;

assert ⊢ extracted_from_longer : typed Nat;

// Test: Verify NonEmptyVector is a subtype of VectorSig (constructor subset subtyping)
// This is the KEY relationship from the paper - NonEmptyVector has only 'cons', VectorSig has 'empty' and 'cons'
assert ⊢ subtype (S Z) (S (S Z)) (NonEmptyVector Nat z) (VectorSig Nat (succ z)) : TYPE;
assert ⊢ subtype (S Z) (S (S Z)) (NonEmptyVector Nat (succ z)) (VectorSig Nat (succ (succ z))) : TYPE;

symbol NonZero: typed (signature (S Z));
rule NonZero ↪ mk_signature (pi_append Nat (pi_nul Nat)) succ_base;

// Constructor application: get a constructor from a signature
// The constructor is typed by its kind (from the signature)
symbol mk_constructor : Π (n : nat) (sig : typed (signature (S n))) (i : nat),
  sigma (λ (k : kind), k);  // Returns the constructor with its type

// Get the first constructor (index 0) from a signature_cons
rule mk_constructor $n (signature_cons $n $k $p $rest) Z ↪
  mk_sigma (λ (k : kind), k) $k $p;

// Get a later constructor (index > 0) by recursing into the rest
rule mk_constructor (S $n) (signature_cons (S $n) $k $p $rest) (S $i) ↪
  mk_constructor $n $rest $i;

// NonEmpty signature - similar to NonZero, represents non-zero natural numbers
// Has only the successor constructor
symbol NonEmpty : typed (signature (S Z));
rule NonEmpty ↪ mk_signature (pi_append Nat (pi_nul Nat)) succ_base;

// Test: Verify that NonZero is a subtype of signature_bottom
// The subtype relation should have type TYPE and reduce to unit
assert ⊢ subtype (S Z) Z NonZero signature_bottom : TYPE;

// Test: Get the z constructor from NatSig
// Returns the z constructor with its type (pi_nul Nat)
assert ⊢ mk_constructor (S Z) NatSig Z ≡
  mk_sigma (λ (k : kind), k) (pi_nul Nat) z_base;

// Test: Get the s constructor from NatSig (index 1)
// The result is equivalent to using mk_signature directly
assert ⊢ mk_constructor (S Z) NatSig (S Z) ≡
  mk_sigma (λ (k : kind), k) (pi_append (interpret (S (S Z)) NatSig) (pi_nul Nat)) s;

// Test: Get the s constructor from NonEmpty
// Demonstrates that mk_constructor extracts the same constructor used in mk_signature
assert ⊢ mk_constructor Z NonEmpty Z ≡
  mk_sigma (λ (k : kind), k) (pi_append Nat (pi_nul Nat)) succ_base;

// Test: mk_signature creates the same signature as the explicit signature_cons
assert ⊢ mk_signature (pi_nul Nat) z_base ≡
  signature_cons Z (pi_nul Nat) z_base signature_bottom;

// Test: Verify NonEmpty is a subtype of signature_bottom
assert ⊢ subtype (S Z) Z NonEmpty signature_bottom : TYPE;

// Test: NonEmpty and NonZero should be equivalent (both have same structure)
assert ⊢ NonEmpty ≡ NonZero;

// Extract the s constructor from NonZero and verify it's the same s we use
// This shows that (s z) is constructed using a constructor from NonZero
assert ⊢ mk_constructor Z NonZero Z ≡
  mk_sigma (λ (k : kind), k) (pi_append Nat (pi_nul Nat)) succ_base;

// Similarly for NonEmpty - the s constructor is contained in it
assert ⊢ mk_constructor Z NonEmpty Z ≡
  mk_sigma (λ (k : kind), k) (pi_append Nat (pi_nul Nat)) succ_base;

// Test: interpret preserves signature information - no reduction
// Different signatures create different types
symbol JustZero : typed (signature (S Z));
rule JustZero ↪ mk_signature (pi_nul Nat) z_base;

// Test: Using inject to type values by signatures
// inject now requires specifying the base type (Nat) explicitly
assert ⊢ inject Z NonZero Nat (succ_base z_base) : typed (interpret (S Z) NonZero);
assert ⊢ inject Z JustZero Nat z_base : typed (interpret (S Z) JustZero);
assert ⊢ inject Z NonEmpty Nat (succ_base (succ_base z_base)) : typed (interpret (S Z) NonEmpty);

// Test: Different signatures create distinct types even with same base
// inject Z NonZero Nat (s z) has type typed (interpret (S Z) NonZero)
// inject Z JustZero Nat z has type typed (interpret (S Z) JustZero)
// These are different types that preserve signature information

// Cast function: convert values between signature types based on subtyping
// Given a proof that A <: B, cast a value from type (interpret n A) to (interpret m B)
// This is opaque - no reduction - it's a primitive type coercion operation
symbol cast : Π (n : nat) (m : nat) (A : typed (signature n)) (B : typed (signature m)),
  subtype n m A B → typed (interpret n A) → typed (interpret m B);

// IMPORTANT: Signature-typed values must be created using inject
// Helper functions typed X → typed (interpret ...) are allowed IF they use inject internally

// Convenience helpers for creating NonZero values
symbol mk_nonzero : typed Nat → typed (interpret (S Z) NonZero);
rule mk_nonzero $n ↪ inject Z NonZero Nat (succ_base $n);

// Test: mk_nonzero is a convenience wrapper around inject
assert ⊢ mk_nonzero z_base : typed (interpret (S Z) NonZero);
assert ⊢ mk_nonzero z_base ≡ inject Z NonZero Nat (succ_base z_base);
assert ⊢ mk_nonzero (succ_base z_base) ≡ inject Z NonZero Nat (succ_base (succ_base z_base));

// Test: Creating NatSig values using inject directly
assert ⊢ inject (S Z) NatSig Nat z_base : typed (interpret (S (S Z)) NatSig);
assert ⊢ inject (S Z) NatSig Nat (succ_base z_base) : typed (interpret (S (S Z)) NatSig);
assert ⊢ inject (S Z) NatSig Nat (succ_base (succ_base z_base)) : typed (interpret (S (S Z)) NatSig);

// Test: Creating NonZero values using inject
assert ⊢ inject Z NonZero Nat (succ_base z_base) : typed (interpret (S Z) NonZero);
assert ⊢ inject Z NonZero Nat (succ_base (succ_base z_base)) : typed (interpret (S Z) NonZero);
assert ⊢ inject Z NonZero Nat (succ_base (succ_base (succ_base z_base))) : typed (interpret (S Z) NonZero);

// Example function type using match_interpret:
// This would extract the predecessor from a NonZero value
// The match_cases type for NonZero is: Π (case_s : typed Nat → typed Nat), unit
// So we need to provide:
//   - A lambda for the s constructor case: (λ n, n) to extract the predecessor
//   - unit for the base case (signature_bottom has no constructors)
symbol nonzero_pred : typed (interpret (S Z) NonZero) → typed Nat;

// Example function using match_interpret to increment a NonZero
// Takes a NonZero and returns another NonZero with one more S
symbol nonzero_succ : typed (interpret (S Z) NonZero) → typed (interpret (S Z) NonZero);

// Reduction rules using match_interpret
// For NonZero (1 constructor), match_cases returns just the case type (typed Nat → typed Q)
// So we provide just the lambda, no unit needed
rule nonzero_pred $v ↪
  match_interpret (S Z) NonZero Nat $v
    (λ n, n);  // case for s constructor: return the predecessor

rule nonzero_succ $v ↪
  match_interpret (S Z) NonZero (interpret (S Z) NonZero) $v
    (λ n, inject Z NonZero Nat (succ_base (succ_base n)));  // case for s: wrap (s (s n))

// Test: nonzero_pred extracts predecessor from NonZero values
assert ⊢ nonzero_pred (inject Z NonZero Nat (succ_base z_base)) : typed Nat;
assert ⊢ nonzero_pred (inject Z NonZero Nat (succ_base (succ_base z_base))) : typed Nat;
assert ⊢ nonzero_pred (inject Z NonZero Nat (succ_base (succ_base (succ_base z_base)))) : typed Nat;

// Test: Complete workflow using ONLY inject
// Extract predecessor, apply s, create new NonZero using inject
assert ⊢ inject Z NonZero Nat (succ_base (nonzero_pred (inject Z NonZero Nat (succ_base z_base))))
  : typed (interpret (S Z) NonZero);

// Test: nonzero_succ increments a NonZero
assert ⊢ nonzero_succ (inject Z NonZero Nat (succ_base z_base)) : typed (interpret (S Z) NonZero);
assert ⊢ nonzero_succ (inject Z NonZero Nat (succ_base (succ_base z_base))) : typed (interpret (S Z) NonZero);
assert ⊢ nonzero_succ (inject Z NonZero Nat (succ_base (succ_base (succ_base z_base)))) : typed (interpret (S Z) NonZero);

// Test: Verify that NonZero is a subtype of NatSig
// NonZero has only s constructor, NatSig has both z and s
assert ⊢ subtype (S Z) (S (S Z)) NonZero NatSig : TYPE;

// Test: Cast a value from NonZero to NatSig
// A value typed by NonZero can be used where NatSig is expected
assert ⊢ cast (S Z) (S (S Z)) NonZero NatSig u (inject Z NonZero Nat (succ_base z_base))
  : typed (interpret (S (S Z)) NatSig);

// Test: Cast another NonZero value to NatSig
assert ⊢ cast (S Z) (S (S Z)) NonZero NatSig u (inject Z NonZero Nat (succ_base (succ_base z_base)))
  : typed (interpret (S (S Z)) NatSig);

// Test: NonEmpty is also a subtype of NatSig (since NonEmpty ≡ NonZero)
assert ⊢ subtype (S Z) (S (S Z)) NonEmpty NatSig : TYPE;

// Test: Cast a NonEmpty value to NatSig
assert ⊢ cast (S Z) (S (S Z)) NonEmpty NatSig u (inject Z NonEmpty Nat (succ_base z_base))
  : typed (interpret (S (S Z)) NatSig);

// Test: Cast NonEmptyVector values to VectorSig type using the subtyping relationship
// This demonstrates the KEY contribution from the paper: NonEmptyVector values can be used where VectorSig values are expected
symbol casted_nev : typed (interpret (S (S Z)) (VectorSig Nat (succ z)));
rule casted_nev ↪ cast (S Z) (S (S Z)) (NonEmptyVector Nat z) (VectorSig Nat (succ z)) u nev_0;

assert ⊢ casted_nev : typed (interpret (S (S Z)) (VectorSig Nat (succ z)));

// Test: Cast another NonEmptyVector value
symbol casted_nev_2 : typed (interpret (S (S Z)) (VectorSig Nat (succ z)));
rule casted_nev_2 ↪ cast (S Z) (S (S Z)) (NonEmptyVector Nat z) (VectorSig Nat (succ z)) u nev_2;

assert ⊢ casted_nev_2 : typed (interpret (S (S Z)) (VectorSig Nat (succ z)));
