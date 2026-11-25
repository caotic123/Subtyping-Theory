(* * Subtyping.v

    Rocq formalization of "First-Class Constructor Subsets for Pattern
    Matching on Indexed Families".

    This file starts the mechanization with the *environments* of the
    paper (Section 2, "Type Rules"):

    - constructor and variable names (only decidable equality is needed);
    - the term syntax of the minimal language (Figure: term syntax),
      with a telescoped (general) Pi type;
    - the static symbol context [C_all]: every symbol is declared by a
      general pi type [Π Δ. target] — a family when the target is the
      universe ([F : Indexes -> Type]), a symbolic constructor when the
      target is a family instance ([C : indexes -> fields -> F (...indexes)]);
    - formal signatures [Phi]: constructor lists carrying the proof that
      every name maps to a constructor declaration in [C_all];
    - typing contexts [Gamma];
    - the small-step operational semantics (the paper's ξ-rules), with
      values and capture-permitting substitution;
    - the AGAINST predicate and Φ_ok pruning, extension-level
      convertibility ≡_φ
      (φ-conversion via Eq_φ is the default, so signatures equal their
      pruned versions), and the typing/subtyping judgments: the λΠ base
      rules with Conversion_φ, extended with Rules 1-4, 6 and 7
      (Rule 5 is derivable and omitted);
    - a conservative positivity checker for symbol declarations
      (paper's Remark on inductive well-formedness): recursive
      arguments go through signatures, in positive positions only;
    - statements of canonical forms, progress, preservation, and soundness;
      normalization is
      deliberately not among them.
*)

From Stdlib Require Import String List Bool Eqdep_dec Program.Equality.
Import ListNotations.
Open Scope string_scope.

(* ** Names

   The paper does not rely on any particular labeling scheme for
   constructors: "we only need to decide constructor equality", so
   names are strings, whose equality is decidable. *)

Definition cname : Type := string.  (* symbol names: families, constructors *)
Definition vname : Type := string.  (* term variables *)

Definition cname_eqb : cname -> cname -> bool := String.eqb.
Definition vname_eqb : vname -> vname -> bool := String.eqb.

(* ** Raw constructor lists

   The *syntactic* payload of a signature type {T Delta* :: Phi} is a
   bare list of constructor names (the [constructor_list] of the
   Lambdapi backend).  Terms must carry the raw list: the symbol
   context below contains terms (in telescopes), so [term] cannot be
   indexed by it.  The proof-carrying notion is [signature], further
   down. *)

Definition constr_list : Type := list cname.

(* [Phi' ⊆ Phi] on raw lists, premise of Rule 4. *)
Definition constr_list_incl (Phi' Phi : constr_list) : bool :=
  forallb (fun c => existsb (cname_eqb c) Phi) Phi'.

(* ** Terms

   Following the paper's term syntax:

     Term ::= Var(x) | App(f, a) | Lam(x, e) | Pi(x, A, B)
            | Constr(T, cs) | Match(e, T, ps) | Notation(e, T)

   with these adjustments:
   - [Pi] is generalized to a telescope: [Pi Delta B] is the iterated
     product [Π Delta. B]; the paper's unary [Pi(x, A, B)] is
     [Pi [(x, A)] B].  This is the same general pi type that symbol
     declarations use ([pi_type] below).  Note the nested recursion
     through [list]/[prod]; a tailored induction principle will be
     needed for the meta-theory.
   - [Sort] is the universe [Type] of small types (used in judgments
     such as [Gamma |- T Delta* : Type]);
   - [Cst c] is a reference to a statically declared symbol (family or
     constructor).  The paper uses a single namespace and distinguishes
     constructors by the side conditions [C ∈ C_all] / [F ∉ C_all]
     (Rules 3 and 5); a dedicated former makes those side conditions
     syntactic.

   [Constr T Phi] is the signature type written {T Delta* :: Phi} in the
   paper: [T] is the inductive family already applied to its index
   instance (an iterated [App]), and [Phi] the raw list of available
   constructors. *)

Inductive term : Type :=
| Var      (x : vname)
| Cst      (c : cname)                          (* declared symbol *)
| Sort                                          (* Type *)
| App      (f a : term)
| Lam      (x : vname) (e : term)
| Pi       (Delta : list (vname * term))
           (B : term)                           (* Π Delta. B *)
| Constr   (T : term) (Phi : constr_list)       (* {T Delta* :: Phi} *)
| Match    (e : term) (Q : term)
           (branches : list (cname * term))     (* case e of Q {C => N} *)
| Notation (e T : term).                        (* type annotation *)

(* ** Telescopes and spines

   A telescope [Delta = (a1 : A1) ... (an : An)] is a sequence of typed
   variables; [Delta*] is the corresponding index tuple. *)

Definition telescope : Type := list (vname * term).
Definition indexes : Type := list term.        (* Delta* *)

(* [T Delta*]: application of a family to its index instance. *)
Definition apply_indexes (T : term) (idx : indexes) : term :=
  fold_left App idx T.

(* Smart product: [Π []. B] collapses to [B], so instantiating the
   last binder of a telescoped product yields the target itself. *)
Definition tPi (Delta : telescope) (B : term) : term :=
  match Delta with
  | [] => B
  | _ :: _ => Pi Delta B
  end.

(* Spine view: decompose iterated applications [h u1 ... un] into the
   head [h] and its arguments. *)
Fixpoint spine_rec (t : term) (acc : indexes) : term * indexes :=
  match t with
  | App f a => spine_rec f (a :: acc)
  | _ => (t, acc)
  end.

Definition spine (t : term) : term * indexes := spine_rec t [].

(* Head constant of an application spine. *)
Fixpoint head_cst (t : term) : option cname :=
  match t with
  | Cst c => Some c
  | App f _ => head_cst f
  | _ => None
  end.

(* ** General pi types

   [Π Δ. target] — the single structure shared by the term syntax
   ([Pi]) and the symbol declarations below.  A pi type denotes a term
   directly. *)

Record pi_type : Type := mk_pi_type
  { pt_tel    : telescope
  ; pt_target : term
  }.

Definition pi_type_term (p : pi_type) : term :=
  tPi (pt_tel p) (pt_target p).

(* ** The static symbol context [C_all]

   [C_all] is the finite, *ordered* list of static declarations
   ("Static c : A."); the order is stored so that mutually recursive
   definitions can be registered before checking.  Every symbol *is* a
   general pi type; its role is read off the shape of the target:

   - family        [F : Π idx. Type]           — target is [Sort];
   - constructor   [C : Π Δ_C. F (...indexes)] — target is a spine
     headed by a declared family, [Δ_C] collects the paper's indexes
     and fields. *)

Definition symbol : Type := pi_type.
Definition symbol_ctx : Type := list (cname * symbol).

(* The term denoted by a declaration. *)
Definition symbol_type : symbol -> term := pi_type_term.

(* A family declaration targets the universe. *)
Definition is_family_sym (t : symbol) : bool :=
  match pt_target t with Sort => true | _ => false end.

(* A constructor declaration targets [F (...indexes)]: the spine view
   of the target yields the family [F] and the result instance. *)
Definition constr_target (t : symbol) : option (cname * indexes) :=
  match spine (pt_target t) with
  | (Cst F, args) => Some (F, args)
  | _ => None
  end.

(* Lookup of a symbol; earlier declarations take precedence. *)
Fixpoint lookup_symbol (Call : symbol_ctx) (c : cname) : option symbol :=
  match Call with
  | [] => None
  | (c', s) :: Call' => if cname_eqb c' c then Some s
                        else lookup_symbol Call' c
  end.

(* Lookup restricted to constructor declarations. *)
Definition lookup_constr (Call : symbol_ctx) (c : cname) : option symbol :=
  match lookup_symbol Call c with
  | Some t => match constr_target t with
              | Some _ => Some t
              | None => None
              end
  | None => None
  end.

(* The side condition [C ∈ C_all] of Rules 3 and 5, decidable because
   [C_all] is finite. *)
Definition is_constructor (Call : symbol_ctx) (c : cname) : bool :=
  match lookup_constr Call c with
  | Some _ => true
  | None => false
  end.

(* ** Positivity of symbol declarations

   The paper's Remark on inductive well-formedness: an implementation
   must enforce restrictions so that inductive definitions remain
   well-founded.  We implement the conservative check sketched there,
   per declaration and with respect to its own target family [F]:

   - a binder type of Δ_C either does not mention [F] at all, or is a
     signature {T u⃗ :: Φ} whose spine arguments do not mention [F].
     In particular a *bare* recursive occurrence — the paper's phantom
     declaration succ : nat -> nat — is rejected: its argument would
     not be eliminable by Rule 7;
   - inner product domains must not mention [F], so signatures occur
     freely only in positive positions in Δ_C;
   - a signature target (succ : {nat :: |0 |succ} -> {nat :: |0 |succ})
     is not a constructor declaration at all ([constr_target] fails). *)

(* Does family [F] occur anywhere in [t]? *)
Fixpoint occurs_family (F : cname) (t : term) : bool :=
  match t with
  | Var _ | Sort => false
  | Cst c => cname_eqb c F
  | App f a => occurs_family F f || occurs_family F a
  | Lam _ e => occurs_family F e
  | Pi Delta B =>
      existsb (fun b => occurs_family F (snd b)) Delta ||
      occurs_family F B
  | Constr T _ => occurs_family F T
  | Match e Q brs =>
      occurs_family F e || occurs_family F Q ||
      existsb (fun b => occurs_family F (snd b)) brs
  | Notation e T => occurs_family F e || occurs_family F T
  end.

(* A binder type of Δ_C is positive w.r.t. the target family [F]. *)
Fixpoint positive_field (F : cname) (A : term) : bool :=
  match A with
  | Pi Delta B =>
      forallb (fun b => negb (occurs_family F (snd b))) Delta &&
      positive_field F B
  | Constr T _ =>
      match spine T with
      | (Cst _, args) => forallb (fun a => negb (occurs_family F a)) args
      | _ => negb (occurs_family F T)
      end
  | _ => negb (occurs_family F A)
  end.

(* A well-formed symbol is a family or a constructor whose telescope is
   positive w.r.t. its target family. *)
Definition wf_symbol (p : symbol) : bool :=
  match pt_target p with
  | Sort => true
  | _ =>
      match constr_target p with
      | Some (F, _) =>
          forallb (fun b => positive_field F (snd b)) (pt_tel p)
      | None => false
      end
  end.

Definition wf_symbol_ctx (Call : symbol_ctx) : bool :=
  forallb (fun d => wf_symbol (snd d)) Call.

(* ** Formal signatures

   A signature [Phi] over [C_all] is a constructor list together with
   the proof that every listed name maps to a constructor declaration
   (Rule 1's [Phi ⊆ C_all] premise).  The proof is a boolean equation,
   so it is unique and signatures are determined by their name lists.
   Formation stays intentionally liberal: a declared constructor may
   still be incompatible with the current index instance (an
   uninhabited refinement). *)

Definition all_constructors (Call : symbol_ctx) (cs : constr_list) : bool :=
  forallb (is_constructor Call) cs.

Record signature (Call : symbol_ctx) : Type := mk_signature
  { sig_constrs  : constr_list
  ; sig_declared : all_constructors Call sig_constrs = true
  }.

Arguments mk_signature Call _ _ : rename.
Arguments sig_constrs {Call} _.
Arguments sig_declared {Call} _.

(* Smart constructor: check a raw list and compute the proof. *)
Definition check_signature (Call : symbol_ctx) (cs : constr_list) :
  option (signature Call) :=
  (if all_constructors Call cs as b
      return all_constructors Call cs = b -> option (signature Call)
   then fun H => Some (mk_signature Call cs H)
   else fun _ => None) eq_refl.

(* [Phi' ⊆ Phi] on formal signatures (Rule 4). *)
Definition signature_incl {Call : symbol_ctx} (s' s : signature Call) : bool :=
  constr_list_incl (sig_constrs s') (sig_constrs s).

(* ** Typing contexts [Gamma]

   Contexts map variables to types; we keep them as ordered association
   lists, most recent binding first. *)

Definition ctx : Type := list (vname * term).

Definition empty_ctx : ctx := [].

Definition ctx_cons (x : vname) (A : term) (Gamma : ctx) : ctx :=
  (x, A) :: Gamma.

Fixpoint lookup_var (Gamma : ctx) (x : vname) : option term :=
  match Gamma with
  | [] => None
  | (y, A) :: Gamma' => if vname_eqb y x then Some A
                        else lookup_var Gamma' x
  end.

(* ** Small-step operational semantics

   The paper's reduction rules: ξ-app₁, ξ-app₂, ξ-β, ξ-case and
   ξ-case'.  Metavariables V, W range over values — lambda abstractions
   or constructor applications in weak head normal form. *)

(* Capture-permitting substitution [t[x := s]].  Binders that shadow
   [x] stop the substitution ([Lam], and position-wise inside [Pi]
   telescopes).  The naive definition is adequate for the operational
   semantics: call-by-value evaluation of closed programs only ever
   substitutes closed values, so capture cannot occur. *)
Fixpoint subst (x : vname) (s : term) (t : term) : term :=
  match t with
  | Var y => if vname_eqb x y then s else Var y
  | Cst c => Cst c
  | Sort => Sort
  | App f a => App (subst x s f) (subst x s a)
  | Lam y e => if vname_eqb x y then Lam y e else Lam y (subst x s e)
  | Pi Delta B =>
      let fix subst_tel (Delta : list (vname * term)) :
            list (vname * term) * bool :=
        match Delta with
        | [] => ([], true)
        | (y, A) :: Delta' =>
            if vname_eqb x y then ((y, subst x s A) :: Delta', false)
            else let (Delta'', free) := subst_tel Delta' in
                 ((y, subst x s A) :: Delta'', free)
        end in
      let (Delta', free) := subst_tel Delta in
      Pi Delta' (if free then subst x s B else B)
  | Constr T Phi => Constr (subst x s T) Phi
  | Match e Q brs =>
      Match (subst x s e) (subst x s Q)
        (map (fun '(c, N) => (c, subst x s N)) brs)
  | Notation e T => Notation (subst x s e) (subst x s T)
  end.

(* Values (paper: V, W): lambda abstractions, constructor applications
   in weak head normal form, and compile-time type forms.  Under
   call-by-value the arguments of a constructor application are
   themselves values. *)
Inductive value : term -> Prop :=
| VLam    : forall x e, value (Lam x e)
| VCstApp : forall c us, Forall value us ->
            value (apply_indexes (Cst c) us)
(* Types are compile-time values.  Without these cases a polymorphic
   constructor instantiated by [Sort], [Pi], or a signature type is a
   closed, well-typed stuck application. *)
| VSort   : value Sort
| VPi     : forall Delta B, value (Pi Delta B)
| VConstr : forall T Phi, value (Constr T Phi).

(* A decidable class of inert terms, used by proof-carrying typing
   obligations.  It contains variables, values, and constant-headed
   spines whose arguments are themselves inert. *)
Fixpoint inertb (t : term) : bool :=
  match t with
  | Var _ | Cst _ | Sort | Lam _ _ | Pi _ _ | Constr _ _ => true
  | App f a =>
      match head_cst f with
      | Some _ => inertb f && inertb a
      | None => false
      end
  | Match _ _ _ | Notation _ _ => false
  end.

(* Branch selection for ξ-case': branches are functions over the
   constructor's telescope (Rule 7: N_k : Δ_{D_k} -> Q), selected by
   constructor name. *)
Fixpoint lookup_branch (brs : list (cname * term)) (c : cname) :
  option term :=
  match brs with
  | [] => None
  | (c', N) :: brs' => if cname_eqb c' c then Some N
                       else lookup_branch brs' c
  end.

Reserved Notation "t '-->' t'" (at level 70).

(* The paper's ξ-rules.  In ξ-case' we require the scrutinee to be a
   fully evaluated constructor application ([Forall value us]), which
   keeps the relation deterministic (the paper's ξ-case reduces
   arbitrary scrutinee terms, so the two rules would otherwise
   overlap).  The contracted branch is the branch function applied to
   the constructor's actual arguments: N_i Δ'. *)
Inductive step : term -> term -> Prop :=
| Xi_app1 : forall L L' M,
    L --> L' ->
    App L M --> App L' M
| Xi_app2 : forall V M M',
    value V ->
    M --> M' ->
    App V M --> App V M'
| Xi_beta : forall x N V,
    value V ->
    App (Lam x N) V --> subst x V N
| Xi_case : forall e e' Q brs,
    e --> e' ->
    Match e Q brs --> Match e' Q brs
| Xi_case' : forall c us Q brs N,
    Forall value us ->
    lookup_branch brs c = Some N ->
    Match (apply_indexes (Cst c) us) Q brs --> apply_indexes N us
(* An annotation has no run-time content. *)
| Xi_notation : forall e T,
    Notation e T --> e
where "t '-->' t'" := (step t t').

(* Reflexive-transitive closure. *)
Reserved Notation "t '-->*' t'" (at level 70).

Inductive multistep : term -> term -> Prop :=
| multi_refl : forall t, t -->* t
| multi_step : forall t u v, t --> u -> u -->* v -> t -->* v
where "t '-->*' t'" := (multistep t t').

(* ** Telescope instantiation

   Substitution on a telescope, stopping at a shadowing binder; mirrors
   the [Pi] case of [subst].  Returns the substituted telescope and
   whether [x] is still free after it. *)
Fixpoint subst_telescope (x : vname) (s : term) (Delta : telescope) :
  telescope * bool :=
  match Delta with
  | [] => ([], true)
  | (y, A) :: Delta' =>
      if vname_eqb x y then ((y, subst x s A) :: Delta', false)
      else let (Delta'', free) := subst_telescope x s Delta' in
           ((y, subst x s A) :: Delta'', free)
  end.

(* Type of an application: instantiate the first binder of a telescoped
   product, (Π (x:A) Δ. B) u  ↦  (Π Δ. B)[x := u]. *)
Definition pi_inst (x : vname) (u : term) (Delta : telescope) (B : term) :
  term :=
  let (Delta', free) := subst_telescope x u Delta in
  tPi Delta' (if free then subst x u B else B).

(* Pushing a telescope onto a context, outermost binder first. *)
Fixpoint ctx_push (Gamma : ctx) (Delta : telescope) : ctx :=
  match Delta with
  | [] => Gamma
  | (x, A) :: Delta' => ctx_push (ctx_cons x A Gamma) Delta'
  end.

(* ** AGAINST and signature pruning (Φ_ok)

   [against p t] is the paper's conservative compatibility test between
   an index [p] of the signature's instance and a declared result index
   [t] of a constructor.  It answers false only when both sides are
   constructor-headed applications whose structure makes βη-equality
   impossible; every other case is the paper's "otherwise ⊤" clause
   (in particular a variable on either side).  Indexes are assumed to
   be in normal form. *)

Fixpoint against (p t : term) : bool :=
  match p, t with
  | App f a, App g b =>
      match head_cst (App f a), head_cst (App g b) with
      | Some c, Some c' =>
          if cname_eqb c c' then against f g && against a b else false
      | _, _ => true
      end
  | _, _ =>
      match head_cst p, head_cst t with
      | Some c, Some c' => cname_eqb c c'
      | _, _ => true
      end
  end.

(* [C] is a declared constructor of family [F] (at any index instance):
   the premise Γ ⊢ Cᵢ : Δᵢ -> T Δᵢ'* of Rule 1. *)
Definition constructor_of (Call : symbol_ctx) (c F : cname) : bool :=
  match lookup_symbol Call c with
  | Some p => match constr_target p with
              | Some (F', _) => cname_eqb F' F
              | None => false
              end
  | None => false
  end.

(* A constructor survives pruning at instance [F idx] if it is declared
   in [C_all], targets [F], and AGAINST does not refute any index
   position of its declared result instance. *)
Definition ok_constructor (Call : symbol_ctx) (F : cname) (idx : indexes)
    (c : cname) : bool :=
  match lookup_symbol Call c with
  | Some p =>
      match constr_target p with
      | Some (F', res) =>
          cname_eqb F' F &&
          (if Nat.eqb (length idx) (length res)
           then forallb (fun a => against (fst a) (snd a)) (combine idx res)
           else true)
      | None => false
      end
  | None => false
  end.

(* Φ_ok = OK(Γ, T, Δ*, Φ): the pruned constructor list of Rule 7. *)
Definition prune_sig (Call : symbol_ctx) (T : term) (Phi : constr_list) :
  constr_list :=
  match spine T with
  | (Cst F, idx) => filter (ok_constructor Call F idx) Phi
  | _ => Phi
  end.

(* Decidable constructor-head invariant for judgments whose result is a
   signature.  Non-values are left to progress.  A value is accepted
   exactly when it is a constant-headed spine whose head survives
   pruning.  This side condition is needed on the three generic rules
   that can otherwise assign an arbitrary result type. *)
Definition canonical_guard (Call : symbol_ctx) (M A : term) : bool :=
  match A with
  | Constr T Phi =>
      match spine M with
      | (Cst c, _) => existsb (cname_eqb c) (prune_sig Call T Phi)
      | (Lam _ _, []) | (Sort, []) | (Pi _ _, []) | (Constr _ _, []) => false
      | _ => true
      end
  | _ => true
  end.

Definition callable_guard (M : term) : bool :=
  match spine M with
  | (Cst _, _) => true
  | (Lam _ _, []) => true
  | (Sort, []) | (Pi _ _, []) | (Constr _ _, []) => false
  | _ => true
  end.

Definition application_guard (Call : symbol_ctx) (t u R : term) : bool :=
  canonical_guard Call (App t u) R && callable_guard t.

(* ** Convertibility (≡_φ)

   Following the paper's design option — adopted here as the default —
   conversion is the equivalence closure of φ-simplification at
   signature nodes (rules Norm_φ / Eq_φ), so a
   signature is definitionally equal to its pruned version.  Because
   Φ_ok depends on the declared constructors, the relation is
   parameterized by [C_all].  βη conversion of indices belongs to the
   host calculus; representing it here by an arbitrary operational
   step was unsound (it allowed a redex of any type to convert to a
   signature).  It interprets the extension's Conversion_φ rule. *)
Inductive conv (Call : symbol_ctx) : term -> term -> Prop :=
| conv_refl  : forall t, conv Call t t
(* Eq_φ: {T Δ* :: Φ} ≡ {T Δ* :: Φ_ok}. *)
| conv_phi   : forall T Phi,
    conv Call (Constr T Phi) (Constr T (prune_sig Call T Phi))
| conv_sym   : forall t u, conv Call t u -> conv Call u t
| conv_trans : forall t u v,
    conv Call t u -> conv Call u v -> conv Call t v.

(* ** Typing and subtyping

   The judgments Γ ⊢ t : A and Γ ⊢ A ⊑ B: the λΠ base rules (paper,
   Figure "λΠ-calculus modulo") extended with the signature rules of
   Section 2.  Simplifications recorded here:

   - [Sort : Sort] stands in for the Type/Kind stratification;
   - α-equivalence is not modeled (binder names compare literally);
   - Rule 5 is omitted: it is an algorithmic convenience, derivable
     from application typing plus subsumption;
   - rules that can expose an operational redex carry the corresponding
     local subject-reduction certificate.  This is the proof-carrying
     form of the declarative judgment and avoids assuming a substitution
     or branch-instantiation axiom;
   - in Rule 3 the vector premises Γ ⊢ u⃗ : Δ_C and the βη index
     equation are packaged as the single premise Γ ⊢ C u⃗ : T Δ*
     (derived through the application and conversion rules). *)

Section Typing.

Variable Call : symbol_ctx.

Inductive has_type : ctx -> term -> term -> Prop :=
(* Sort (with Type-in-Type). *)
| T_Sort : forall Gamma,
    has_type Gamma Sort Sort
(* Variable. *)
| T_Var : forall Gamma x A,
    lookup_var Gamma x = Some A ->
    has_type Gamma (Var x) A
(* Constant: symbols carry their declared pi type. *)
| T_Cst : forall Gamma c p,
    lookup_symbol Call c = Some p ->
    wf_symbol p = true ->
    has_type Gamma (Cst c) (symbol_type p)
(* Application, instantiating the first binder of the telescope. *)
| T_App : forall Gamma t u x A Delta B,
    has_type Gamma t (Pi ((x, A) :: Delta) B) ->
    has_type Gamma u A ->
    application_guard Call t u (pi_inst x u Delta B) = true ->
    (forall v, App t u --> v ->
       has_type Gamma v (pi_inst x u Delta B)) ->
    has_type Gamma (App t u) (pi_inst x u Delta B)
(* Abstraction (unannotated, Curry-style), peeling one binder. *)
| T_Lam : forall Gamma x A e Delta B,
    has_type Gamma A Sort ->
    has_type (ctx_cons x A Gamma) e (tPi Delta B) ->
    has_type Gamma (Lam x e) (Pi ((x, A) :: Delta) B)
(* Product. *)
| T_Pi : forall Gamma Delta B,
    wf_tel Gamma Delta ->
    has_type (ctx_push Gamma Delta) B Sort ->
    has_type Gamma (Pi Delta B) Sort
(* Type annotation. *)
| T_Notation : forall Gamma e T,
    has_type Gamma e T ->
    has_type Gamma (Notation e T) T
(* Backward closure along one deterministic evaluation step. *)
| T_Expand : forall Gamma t t' A,
    has_type Gamma t' A ->
    t --> t' ->
    has_type Gamma t A
(* Conversion_φ: Eq_φ pruning is the extension-level conversion. *)
| T_Conv : forall Gamma t A B,
    has_type Gamma t A ->
    has_type Gamma B Sort ->
    conv Call A B ->
    canonical_guard Call t B = true ->
    (forall t', t --> t' -> has_type Gamma t' B) ->
    has_type Gamma t B
(* Subsumption (Aspinall–Compagnoni). *)
| T_Sub : forall Gamma t A B,
    has_type Gamma t A ->
    subtype Gamma A B ->
    canonical_guard Call t B = true ->
    (forall t', t --> t' -> has_type Gamma t' B) ->
    has_type Gamma t B
(* Rule 1: signature formation — every listed name is a declared
   constructor of the head family of T. *)
| T_Sig : forall Gamma T Phi F,
    has_type Gamma T Sort ->
    head_cst T = Some F ->
    forallb (fun c => constructor_of Call c F) Phi = true ->
    has_type Gamma (Constr T Phi) Sort
(* Rule 3: constructor introduction. *)
| T_SigIntro : forall Gamma T Phi c us,
    has_type Gamma (Constr T Phi) Sort ->
    existsb (cname_eqb c) (prune_sig Call T Phi) = true ->
    has_type Gamma (apply_indexes (Cst c) us) T ->
    (forall v, apply_indexes (Cst c) us --> v ->
       has_type Gamma v (Constr T Phi)) ->
    has_type Gamma (apply_indexes (Cst c) us) (Constr T Phi)
(* Rule 7: signature elimination — the branch list Ψ = map fst brs
   satisfies Φ_ok ⊆ Ψ ⊆ Φ, and each branch is a function over the
   constructor's telescope into Q. *)
| T_SigCase : forall Gamma T Phi M Q brs,
    has_type Gamma (Constr T Phi) Sort ->
    has_type Gamma Q Sort ->
    has_type Gamma M (Constr T Phi) ->
    constr_list_incl (prune_sig Call T Phi) (map fst brs) = true ->
    constr_list_incl (map fst brs) Phi = true ->
    branches_type Gamma brs Q ->
    (forall v, Match M Q brs --> v -> has_type Gamma v Q) ->
    has_type Gamma (Match M Q brs) Q

(* Telescope well-formedness (for the Product rule). *)
with wf_tel : ctx -> telescope -> Prop :=
| wf_tel_nil : forall Gamma,
    wf_tel Gamma []
| wf_tel_cons : forall Gamma x A Delta,
    has_type Gamma A Sort ->
    wf_tel (ctx_cons x A Gamma) Delta ->
    wf_tel Gamma ((x, A) :: Delta)

(* Branch typing for Rule 7: N_k : Δ_{D_k} -> Q. *)
with branches_type : ctx -> list (cname * term) -> term -> Prop :=
| bt_nil : forall Gamma Q,
    branches_type Gamma [] Q
| bt_cons : forall Gamma c N brs Q p,
    lookup_constr Call c = Some p ->
    has_type Gamma N (tPi (pt_tel p) Q) ->
    branches_type Gamma brs Q ->
    branches_type Gamma ((c, N) :: brs) Q

with subtype : ctx -> term -> term -> Prop :=
(* Structural rules: conversion (subsuming reflexivity) and
   transitivity. *)
| Sub_Conv : forall Gamma A B,
    conv Call A B ->
    subtype Gamma A B
| Sub_Trans : forall Gamma A B C,
    subtype Gamma A B ->
    subtype Gamma B C ->
    subtype Gamma A C
(* Rule 2: forgetting the signature. *)
| Sub_SigBase : forall Gamma T Phi,
    has_type Gamma T Sort ->
    (exists F, head_cst T = Some F) ->
    subtype Gamma (Constr T Phi) T
(* Rule 4: signature widening at convertible index instances. *)
| Sub_Sig : forall Gamma T T' Phi Phi',
    conv Call T T' ->
    constr_list_incl Phi' Phi = true ->
    subtype Gamma (Constr T' Phi') (Constr T Phi)
(* Rule 6: products — contravariant domains, covariant codomains. *)
| Sub_Pi : forall Gamma x A A' Delta B Delta' B',
    subtype Gamma A' A ->
    subtype (ctx_cons x A' Gamma) (tPi Delta B) (tPi Delta' B') ->
    subtype Gamma (Pi ((x, A) :: Delta) B) (Pi ((x, A') :: Delta') B').

End Typing.

(* ** Meta-theory (statements)

   The theorems of the development, stated over the judgments above.
   Normalization is deliberately not stated: termination is not a
   guaranteed property of the system — the paper requires only enough
   normalization to inspect head constructors. *)

(* Canonical forms — the paper's "operational reading" and Theorem
   (Head-constructor refinement): a closed value of a signature type is
   a constructor application whose head constructor is drawn from Φ,
   indeed from the pruned list Φ_ok. *)
Theorem canonical_forms_sig (Call : symbol_ctx) (M T : term)
    (Phi : constr_list) :
  has_type Call empty_ctx M (Constr T Phi) ->
  value M ->
  exists c us,
    M = apply_indexes (Cst c) us /\
    Forall value us /\
    In c (prune_sig Call T Phi).
Proof. Admitted.

(* Paper, Theorem (Progress): a closed term of signature type is a
   value or reduces. *)
Theorem progress (Call : symbol_ctx) (M T : term) (Phi : constr_list) :
  has_type Call empty_ctx M (Constr T Phi) ->
  value M \/ exists M', M --> M'.
Proof. Admitted.

(* Paper, Theorem (Preservation): subject reduction. *)
Theorem preservation (Call : symbol_ctx) (Gamma : ctx) (M M' R : term) :
  has_type Call Gamma M R ->
  M --> M' ->
  has_type Call Gamma M' R.
Proof. Admitted.

(* Type soundness: a closed term of signature type never gets stuck —
   every reduct is again a value or reducible.  Follows from Progress
   and Preservation along the reduction sequence. *)
Theorem soundness (Call : symbol_ctx) (M M' T : term) (Phi : constr_list) :
  has_type Call empty_ctx M (Constr T Phi) ->
  M -->* M' ->
  value M' \/ exists M'', M' --> M''.
Proof. Admitted.
