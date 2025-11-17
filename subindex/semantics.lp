constant symbol kind : TYPE;
constant symbol set1 : kind;
constant symbol nat : TYPE;
constant symbol Z : nat;
constant symbol S : nat → nat;
symbol typed : kind → TYPE;
constant symbol unit : TYPE;
constant symbol u : unit;
constant symbol indexes_type : TYPE;
symbol static_symbol : indexes_type → kind;
symbol arrow : kind → kind → kind;
constant symbol sigma : Π (P : Π (k: kind), kind), TYPE;
constant symbol mk_sigma : Π (P : Π (k: kind), kind) (k : kind), typed (P k) → (sigma P);

constant symbol index_null : indexes_type;
constant symbol index_succ : kind → indexes_type → indexes_type;

rule (typed (static_symbol (index_null))) ↪ kind;
rule (typed (static_symbol (index_succ $k $res))) ↪ Π (n : (typed $k)), (typed (static_symbol $res));

symbol Constructors : Π (k: kind), typed k → typed k → unit;
constant symbol signature : nat → kind;
constant symbol signature_bottom : typed (signature Z);
constant symbol signature_cons
  : Π (n : nat) (k : kind), typed k → typed (signature n) → typed (signature (S n));

symbol constructor_null : Π (k : kind), kind;
symbol constructor_append : kind → kind → kind;

rule typed (constructor_null $k) ↪ (typed $k);
rule typed (constructor_append $k $a) ↪ Π (n : (typed $k)), (typed $a);

// set to return the signature type, and if there is a constructor by lookup the index
symbol choose : Π (n : nat), typed (signature n) → nat → (sigma (λ (k : kind), k));
rule choose (S $n) (signature_cons $n $k $p _) Z ↪ (mk_sigma (λ (k : kind), k) $k $p);
rule choose (S $n) (signature_cons $n _ _ $rs) (S $i) ↪ choose $n $rs $i;

symbol Nat : typed (static_symbol index_null);
constant symbol z : typed Nat;
rule (Constructors _ z z) ↪ u;
constant symbol s : typed Nat → typed Nat;
rule (Constructors _ (s _) (s _)) ↪ u;


// Inductive JMeq { A : Type } (a : A) : forall { B : Type }, B -> Type :=

symbol subtyping: Π (A :kind) (B : kind) (a : typed A), kind;
rule (subtyping $k $k' $a) ↪ (subtyping $k $k $a);

rule (subtyping $k $k $a) ↪ $a;


rule force: Π (k : kind) (k' : kind) (a : typed A), subtyping k k' → 

symbol Vector : typed (static_symbol (index_succ Nat index_null));
constant symbol empty : typed (Vector z);
rule (Constructors _ (empty _) (empty _)) ↪ u;
constant symbol cons: Π (n : typed Nat), typed (Vector n) → typed (Vector (s n));
rule (Constructors _ (cons _) (cons _)) ↪ u;

symbol NonZero: typed (signature (S Z)); 
rule NonZero ↪ signature_cons Z (constructor_append Nat (constructor_null Nat)) s signature_bottom;
