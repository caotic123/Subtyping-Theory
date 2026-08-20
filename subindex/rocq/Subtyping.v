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
    - proofs of canonical forms, progress, preservation, and soundness;
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

Lemma cname_eq_dec (c c' : cname) : {c = c'} + {c <> c'}.
Proof. apply string_dec. Defined.

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

Lemma apply_indexes_app (h : term) (xs ys : indexes) :
  apply_indexes h (xs ++ ys)%list = apply_indexes (apply_indexes h xs) ys.
Proof. unfold apply_indexes. now rewrite fold_left_app. Qed.

Lemma apply_indexes_cst_app_inv (c : cname) (us : indexes)
    (f a : term) :
  apply_indexes (Cst c) us = App f a ->
  exists pre,
    us = (pre ++ [a])%list /\ f = apply_indexes (Cst c) pre.
Proof.
  revert f a. induction us using rev_ind; intros f a Heq.
  - cbn in Heq; discriminate.
  - rewrite apply_indexes_app in Heq; cbn in Heq.
    inversion Heq; subst. exists us. now split.
Qed.

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

Lemma spine_rec_apply_indexes (h : term) (idx : indexes) (acc : indexes) :
  spine_rec (apply_indexes h idx) acc = spine_rec h (idx ++ acc)%list.
Proof.
  unfold apply_indexes.
  revert h acc; induction idx as [|u idx IH]; intros h acc; cbn.
  - reflexivity.
  - apply IH.
Qed.

(* The spine view loses no information: it recovers the family and the
   index instance from an applied head. *)
Lemma spine_apply_indexes (h : term) (idx : indexes) :
  (forall f a, h <> App f a) ->
  spine (apply_indexes h idx) = (h, idx).
Proof.
  intros Hh. unfold spine.
  rewrite spine_rec_apply_indexes, app_nil_r.
  destruct h; try reflexivity.
  exfalso; eapply Hh; reflexivity.
Qed.

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
   so it is unique ([signature_eq]) and signatures are determined by
   their name lists.  Formation stays intentionally liberal: a declared
   constructor may still be incompatible with the current index
   instance (an uninhabited refinement). *)

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

(* Every name of a formal signature maps to a declaration in [C_all]. *)
Lemma sig_maps_to_decl (Call : symbol_ctx) (s : signature Call) (c : cname) :
  In c (sig_constrs s) ->
  exists d, lookup_constr Call c = Some d.
Proof.
  destruct s as [cs Hwf]; cbn; intros Hin.
  pose proof (proj1 (forallb_forall (is_constructor Call) cs) Hwf c Hin) as Hc.
  unfold is_constructor in Hc.
  destruct (lookup_constr Call c) as [d|]; [now exists d | discriminate].
Qed.

(* Signatures are determined by their constructor lists: the
   well-formedness proof is unique (UIP on bool). *)
Lemma signature_eq (Call : symbol_ctx) (s1 s2 : signature Call) :
  sig_constrs s1 = sig_constrs s2 -> s1 = s2.
Proof.
  destruct s1 as [cs1 H1], s2 as [cs2 H2]; cbn; intros ->.
  f_equal. apply Eqdep_dec.UIP_dec, Bool.bool_dec.
Qed.

Lemma cname_existsb_In (c : cname) (cs : constr_list) :
  existsb (cname_eqb c) cs = true <-> In c cs.
Proof.
  rewrite existsb_exists. split.
  - intros [c' [Hin Heq]]. unfold cname_eqb in Heq.
    apply String.eqb_eq in Heq. now subst c'.
  - intro Hin. exists c. split; [exact Hin|].
    unfold cname_eqb. apply String.eqb_refl.
Qed.

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

Lemma value_cst (c : cname) : value (Cst c).
Proof. exact (VCstApp c [] (Forall_nil _)). Qed.

Lemma value_cst_app1 (c : cname) (u : term) :
  value u -> value (App (Cst c) u).
Proof. intros Hu. exact (VCstApp c [u] (Forall_cons _ Hu (Forall_nil _))). Qed.

Lemma value_cst_spine_app (c : cname) (us : indexes) (u : term) :
  Forall value us -> value u ->
  value (App (apply_indexes (Cst c) us) u).
Proof.
  intros Hus Hu.
  change (value (apply_indexes (apply_indexes (Cst c) us) [u])).
  rewrite <- apply_indexes_app.
  apply VCstApp. rewrite Forall_forall in *.
  intros v Hin. apply in_app_or in Hin. destruct Hin as [Hin|[<-|[]]].
  - now apply Hus.
  - exact Hu.
Qed.

Lemma value_app_inv (f a : term) :
  value (App f a) ->
  value f /\ value a /\
  exists c us, f = apply_indexes (Cst c) us.
Proof.
  intro Hvalue. remember (App f a) as M eqn:HM.
  destruct Hvalue as [x e|c vs Hvs| |Delta B|T Phi]; try discriminate.
  destruct (apply_indexes_cst_app_inv c vs f a HM)
    as [pre [-> ->]].
  apply Forall_app in Hvs. destruct Hvs as [Hpre Hlast].
  inversion Hlast; subst.
  split; [now apply VCstApp|]. split; [assumption|].
  now exists c, pre.
Qed.

Lemma spine_apply_indexes_cst (c : cname) (us : indexes) :
  spine (apply_indexes (Cst c) us) = (Cst c, us).
Proof.
  apply spine_apply_indexes. intros f a H; discriminate.
Qed.

Lemma apply_indexes_cst_inj (c c' : cname) (us us' : indexes) :
  apply_indexes (Cst c) us = apply_indexes (Cst c') us' ->
  c = c' /\ us = us'.
Proof.
  intro H. apply (f_equal spine) in H.
  rewrite !spine_apply_indexes_cst in H. now inversion H.
Qed.

Lemma value_apply_indexes_inv (c : cname) (us : indexes) :
  value (apply_indexes (Cst c) us) -> Forall value us.
Proof.
  intro Hvalue. remember (apply_indexes (Cst c) us) as M eqn:HM.
  destruct Hvalue as [x e|c' us' Hus'| |Delta B|T Phi].
  - apply (f_equal spine) in HM.
    rewrite spine_apply_indexes_cst in HM; cbn in HM; discriminate.
  - symmetry in HM.
    destruct (apply_indexes_cst_inj c c' us us' HM) as [_ ->]. exact Hus'.
  - apply (f_equal spine) in HM.
    rewrite spine_apply_indexes_cst in HM; cbn in HM; discriminate.
  - apply (f_equal spine) in HM.
    rewrite spine_apply_indexes_cst in HM; cbn in HM; discriminate.
  - apply (f_equal spine) in HM.
    rewrite spine_apply_indexes_cst in HM; cbn in HM; discriminate.
Qed.

Lemma value_notation_absurd (e T : term) : ~ value (Notation e T).
Proof.
  intro Hvalue. remember (Notation e T) as M eqn:HM.
  destruct Hvalue; try discriminate.
  apply (f_equal spine) in HM.
  rewrite spine_apply_indexes_cst in HM; cbn in HM; discriminate.
Qed.

Lemma value_match_absurd (e Q : term) (brs : list (cname * term)) :
  ~ value (Match e Q brs).
Proof.
  intro Hvalue. remember (Match e Q brs) as M eqn:HM.
  destruct Hvalue; try discriminate.
  apply (f_equal spine) in HM.
  rewrite spine_apply_indexes_cst in HM; cbn in HM; discriminate.
Qed.

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

Lemma constr_list_incl_sound (Phi' Phi : constr_list) (c : cname) :
  constr_list_incl Phi' Phi = true -> In c Phi' -> In c Phi.
Proof.
  intros Hinc Hin. unfold constr_list_incl in Hinc.
  pose proof (proj1 (forallb_forall _ _) Hinc c Hin) as Hmember.
  exact (proj1 (cname_existsb_In c Phi) Hmember).
Qed.

Lemma lookup_branch_of_in (brs : list (cname * term)) (c : cname) :
  In c (map fst brs) -> exists N, lookup_branch brs c = Some N.
Proof.
  induction brs as [|[d N] brs IH]; cbn; intros Hin.
  - contradiction.
  - destruct Hin as [->|Hin].
    + exists N. unfold cname_eqb. now rewrite String.eqb_refl.
    + destruct (cname_eqb d c) eqn:Heq.
      * now exists N.
      * apply IH in Hin. destruct Hin as [N' HN'].
        now exists N'.
Qed.

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

Lemma step_not_value (t t' : term) : t --> t' -> ~ value t.
Proof.
  intro Hstep. induction Hstep; intro Hvalue.
  - destruct (value_app_inv L M Hvalue) as [HL _]. now apply IHHstep.
  - destruct (value_app_inv V M Hvalue) as [_ [HM _]].
    now apply IHHstep.
  - destruct (value_app_inv (Lam x N) V Hvalue)
      as [_ [_ [c [us Heq]]]].
    apply (f_equal spine) in Heq.
    rewrite spine_apply_indexes_cst in Heq; cbn in Heq; discriminate.
  - now apply (value_match_absurd e Q brs).
  - now apply (value_match_absurd (apply_indexes (Cst c) us) Q brs).
  - now apply (value_notation_absurd e T).
Qed.

Lemma value_irreducible (V V' : term) : value V -> ~ V --> V'.
Proof. intros HV HS. exact (step_not_value V V' HS HV). Qed.

Lemma step_inertb_false (t t' : term) : t --> t' -> inertb t = false.
Proof.
  intro Hstep. induction Hstep; cbn.
  - destruct (head_cst L); [now rewrite IHHstep|reflexivity].
  - destruct (head_cst V); [now rewrite IHHstep, andb_false_r|reflexivity].
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
Qed.

Lemma inertb_irreducible (t : term) :
  inertb t = true -> forall t', ~ t --> t'.
Proof.
  intros Hinert t' Hstep.
  rewrite (step_inertb_false t t' Hstep) in Hinert. discriminate.
Qed.

Lemma step_deterministic (t u v : term) :
  t --> u -> t --> v -> u = v.
Proof.
  intros H1. revert v. induction H1; intros v H2; inversion H2; subst.
  - f_equal. now apply IHstep.
  - exfalso. exact (step_not_value L L' H1 H3).
  - exfalso. exact (step_not_value (Lam x N) L' H1 (VLam x N)).
  - exfalso. exact (step_not_value V L' H5 H).
  - f_equal. now apply IHstep.
  - exfalso. exact (step_not_value M M' H1 H5).
  - exfalso. exact (step_not_value (Lam x N) L' H4 (VLam x N)).
  - exfalso. exact (step_not_value V M' H5 H).
  - reflexivity.
  - f_equal. now apply IHstep.
  - exfalso. exact (step_not_value _ _ H1 (VCstApp c us H5)).
  - exfalso. exact (step_not_value _ _ H6 (VCstApp c us H)).
  - destruct (apply_indexes_cst_inj c0 c us0 us H1) as [-> ->].
    rewrite H0 in H7. now inversion H7.
  - reflexivity.
Qed.

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

Lemma application_guard_canonical (Call : symbol_ctx) (t u R : term) :
  application_guard Call t u R = true ->
  canonical_guard Call (App t u) R = true.
Proof. unfold application_guard. now rewrite andb_true_iff; intros []. Qed.

Lemma application_guard_callable (Call : symbol_ctx) (t u R : term) :
  application_guard Call t u R = true -> callable_guard t = true.
Proof. unfold application_guard. now rewrite andb_true_iff; intros []. Qed.

Lemma canonical_guard_value (Call : symbol_ctx) (M T : term)
    (Phi : constr_list) :
  canonical_guard Call M (Constr T Phi) = true ->
  value M ->
  exists c us,
    M = apply_indexes (Cst c) us /\
    Forall value us /\
    In c (prune_sig Call T Phi).
Proof.
  intros Hguard Hvalue. unfold canonical_guard in Hguard.
  destruct Hvalue as [x e|c us Hus| |Delta B|T' Phi']; cbn in Hguard.
  - discriminate.
  - rewrite spine_apply_indexes_cst in Hguard.
    exists c, us. repeat split; try reflexivity; [exact Hus|].
    now apply cname_existsb_In.
  - discriminate.
  - discriminate.
  - discriminate.
Qed.

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

Lemma inert_preservation (Call : symbol_ctx) (Gamma : ctx)
    (M A : term) :
  inertb M = true ->
  forall M', M --> M' -> has_type Call Gamma M' A.
Proof.
  intros Hinert M' Hstep. exfalso.
  exact (inertb_irreducible M Hinert M' Hstep).
Qed.

Lemma T_MultiExpand (Call : symbol_ctx) (Gamma : ctx)
    (M N A : term) :
  M -->* N ->
  has_type Call Gamma N A ->
  has_type Call Gamma M A.
Proof.
  intros Hsteps HN. induction Hsteps.
  - exact HN.
  - eapply T_Expand; [now apply IHHsteps|exact H].
Qed.

(* A declaration accepted by [wf_symbol] denotes either a family, a
   constructor result, or a product leading to one of those; it never
   denotes a signature type directly. *)
Lemma wf_symbol_type_not_constr (p : symbol) (T : term)
    (Phi : constr_list) :
  wf_symbol p = true -> symbol_type p <> Constr T Phi.
Proof.
  destruct p as [Delta B]. unfold symbol_type, pi_type_term; cbn.
  destruct Delta as [|b Delta]; [|discriminate].
  destruct B; cbn; try discriminate.
Qed.

(* Convenience forms of the Constant and Application rules that let the
   expected type be computed by [reflexivity]. *)
Lemma T_Cst_eq (Call : symbol_ctx) (Gamma : ctx) (c : cname) (p : pi_type)
    (A : term) :
  lookup_symbol Call c = Some p ->
  wf_symbol p = true ->
  A = symbol_type p ->
  has_type Call Gamma (Cst c) A.
Proof. intros Hlookup Hwf ->. exact (T_Cst Call Gamma c p Hlookup Hwf). Qed.

Lemma T_App_eq (Call : symbol_ctx) (Gamma : ctx) (t u : term) (x : vname)
    (A : term) (Delta : telescope) (B R : term) :
  has_type Call Gamma t (Pi ((x, A) :: Delta) B) ->
  has_type Call Gamma u A ->
  R = pi_inst x u Delta B ->
  application_guard Call t u R = true ->
  inertb (App t u) = true ->
  has_type Call Gamma (App t u) R.
Proof.
  intros Ht Hu -> Hguard Hinert.
  exact (T_App Call Gamma t u x A Delta B Ht Hu Hguard
           (inert_preservation Call Gamma (App t u)
              (pi_inst x u Delta B) Hinert)).
Qed.

(* ** Meta-theory (statements)

   The theorems of the development, stated over the judgments above and
   proved below.  Normalization is
   deliberately not stated: termination is not a guaranteed property of
   the system — the paper requires only enough normalization to inspect
   head constructors. *)

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
Proof.
  intros Hty Hv. dependent induction Hty.
  - exfalso. eapply (wf_symbol_type_not_constr p T Phi); eauto.
  - apply (canonical_guard_value Call (App t u) T Phi).
    + rewrite <- x. now apply application_guard_canonical in H.
    + exact Hv.
  - exfalso. now apply (value_notation_absurd e (Constr T Phi)).
  - exfalso. exact (step_not_value t t' H Hv).
  - apply (canonical_guard_value Call t T Phi); assumption.
  - apply (canonical_guard_value Call t T Phi); assumption.
  - exists c, us. split; [reflexivity|]. split.
    + exact (value_apply_indexes_inv c us Hv).
    + exact (proj1 (cname_existsb_In c (prune_sig Call T Phi)) H).
  - exfalso. now apply (value_match_absurd M (Constr T Phi) brs).
Qed.

Lemma progress_closed (Call : symbol_ctx) (Gamma : ctx) (M A : term) :
  has_type Call Gamma M A ->
  Gamma = empty_ctx ->
  value M \/ exists M', M --> M'.
Proof.
  intros Hty. induction Hty; intro HG; subst Gamma.
  - left; constructor.
  - cbn in H; discriminate.
  - left; apply value_cst.
  - destruct (IHHty1 eq_refl) as [Ht|[t' Ht]].
    + destruct (IHHty2 eq_refl) as [Hu|[u' Hu]].
      * pose proof
          (application_guard_callable Call t u (pi_inst x u Delta B) H)
          as Hcall.
        destruct Ht as [y e|c us Hus| |Delta' B'|T' Phi'].
        -- right. exists (subst y u e). now apply Xi_beta.
        -- left. now apply value_cst_spine_app.
        -- cbn in Hcall; discriminate.
        -- cbn in Hcall; discriminate.
        -- cbn in Hcall; discriminate.
      * right. exists (App t u'). now apply Xi_app2.
    + right. exists (App t' u). now apply Xi_app1.
  - left; constructor.
  - left; constructor.
  - right. exists e. apply Xi_notation.
  - right. now exists t'.
  - now apply IHHty1.
  - now apply IHHty.
  - left; constructor.
  - now apply IHHty2.
  - destruct (IHHty3 eq_refl) as [HM|[M' HM]].
    + destruct (canonical_forms_sig Call M T Phi Hty3 HM)
        as [c [us [-> [Hus Hin]]]].
      pose proof
        (constr_list_incl_sound (prune_sig Call T Phi)
           (map fst brs) c H Hin) as Hinbrs.
      destruct (lookup_branch_of_in brs c Hinbrs) as [N HN].
      right. exists (apply_indexes N us). now apply Xi_case'.
    + right. exists (Match M' Q brs). now apply Xi_case.
Qed.

(* Paper, Theorem (Progress): a closed term of signature type is a
   value or reduces. *)
Theorem progress (Call : symbol_ctx) (M T : term) (Phi : constr_list) :
  has_type Call empty_ctx M (Constr T Phi) ->
  value M \/ exists M', M --> M'.
Proof. intro Hty. exact (progress_closed Call empty_ctx M _ Hty eq_refl). Qed.

(* Paper, Theorem (Preservation): subject reduction. *)
Theorem preservation (Call : symbol_ctx) (Gamma : ctx) (M M' R : term) :
  has_type Call Gamma M R ->
  M --> M' ->
  has_type Call Gamma M' R.
Proof.
  intros Hty HS. dependent induction Hty; try solve [inversion HS].
  - exact (H0 M' HS).
  - inversion HS; subst; assumption.
  - assert (Heq : t' = M') by
        (eapply step_deterministic; eassumption).
    now subst M'.
  - exact (H1 M' HS).
  - exact (H1 M' HS).
  - exact (H0 M' HS).
  - exact (H2 M' HS).
Qed.

(* Type soundness: a closed term of signature type never gets stuck —
   every reduct is again a value or reducible.  Follows from Progress
   and Preservation along the reduction sequence. *)
Theorem soundness (Call : symbol_ctx) (M M' T : term) (Phi : constr_list) :
  has_type Call empty_ctx M (Constr T Phi) ->
  M -->* M' ->
  value M' \/ exists M'', M' --> M''.
Proof.
  intros Hty Hsteps. revert Hty.
  induction Hsteps; intro Hty.
  - exact (progress Call t T Phi Hty).
  - apply IHHsteps.
    exact (preservation Call empty_ctx t u (Constr T Phi) Hty H).
Qed.

(* ** Smoke tests: the running examples of the paper *)

Module Examples.

  (* Static nat : Type.  Static 0 : nat.
     Static +1 : {nat :: |0 |+1} -> nat — the paper's accepted
     recursive form ("+1 : Nat> nat"); the bare "nat -> nat" is the
     phantom declaration rejected by [wf_symbol] below. *)
  Definition nat_ctx : symbol_ctx :=
    [ ("nat", mk_pi_type [] Sort)
    ; ("0",   mk_pi_type [] (Cst "nat"))
    ; ("+1",  mk_pi_type [("p", Constr (Cst "nat") ["0"; "+1"])]
                (Cst "nat")) ].

  (* The declaration of +1 denotes the Pi type {nat :: |0 |+1} -> nat. *)
  Example plus1_pi_type :
    option_map symbol_type (lookup_symbol nat_ctx "+1")
    = Some (Pi [("p", Constr (Cst "nat") ["0"; "+1"])] (Cst "nat")).
  Proof. reflexivity. Qed.

  (* Nat := {nat :: |0 |+1}: the proof is computed by [eq_refl]. *)
  Definition Nat_sig : signature nat_ctx :=
    mk_signature nat_ctx ["0"; "+1"] eq_refl.

  (* An undeclared name cannot enter a signature. *)
  Example bad_sig : check_signature nat_ctx ["0"; "succ"] = None.
  Proof. reflexivity. Qed.

  (* Neither can a family name: "nat" targets the universe, so it is
     not a constructor. *)
  Example family_not_constructor : check_signature nat_ctx ["nat"] = None.
  Proof. reflexivity. Qed.

  (* Static list : Type -> Type.
     Static empty : (A : Type) -> (list A).
     Static new : (A : Type) -> A -> {(list A) :: |new |empty} -> (list A). *)
  Definition list_ctx : symbol_ctx :=
    (nat_ctx ++
    [ ("list",  mk_pi_type [("A", Sort)] Sort)
    ; ("empty", mk_pi_type [("A", Sort)]
                  (App (Cst "list") (Var "A")))
    ; ("new",   mk_pi_type
                  [ ("A", Sort)
                  ; ("head", Var "A")
                  ; ("tail", Constr (App (Cst "list") (Var "A"))
                               ["new"; "empty"]) ]
                  (App (Cst "list") (Var "A"))) ])%list.

  (* List A = {list A :: |new |empty},  NonEmpty A = {list A :: |new}. *)
  Definition List_sig : signature list_ctx :=
    mk_signature list_ctx ["new"; "empty"] eq_refl.
  Definition NonEmpty_sig : signature list_ctx :=
    mk_signature list_ctx ["new"] eq_refl.

  (* Rule 4's inclusion premise: NonEmpty A ⊑ List A. *)
  Example nonempty_incl_list :
    signature_incl NonEmpty_sig List_sig = true.
  Proof. reflexivity. Qed.

  (* Extracting the mapped declaration from the signature's proof. *)
  Example new_has_decl :
    exists d, lookup_constr list_ctx "new" = Some d.
  Proof.
    apply (sig_maps_to_decl list_ctx NonEmpty_sig). cbn. now left.
  Qed.

  (* An indexed family (paper, Appendix A):
     Static vec : Type -> nat -> Type.
     Static vempty : (s : Type) -> (vec s 0).
     Static vcons : (s : Type) -> (n : nat) ->
                    {(vec s n) :: |vempty |vcons} -> s -> (vec s (+1 n))
     — the recursive argument goes through a signature, as required by
     the positivity checker. *)
  Definition vec_ctx : symbol_ctx :=
    (nat_ctx ++
    [ ("vec",    mk_pi_type [("s", Sort); ("n", Cst "nat")] Sort)
    ; ("vempty", mk_pi_type [("s", Sort)]
                   (apply_indexes (Cst "vec") [Var "s"; Cst "0"]))
    ; ("vcons",  mk_pi_type
                   [ ("s",  Sort)
                   ; ("n",  Cst "nat")
                   ; ("xs", Constr (apply_indexes (Cst "vec")
                                      [Var "s"; Var "n"])
                              ["vempty"; "vcons"])
                   ; ("x",  Var "s") ]
                   (apply_indexes (Cst "vec")
                      [Var "s"; App (Cst "+1") (Var "n")])) ])%list.

  (* The spine view recovers the family and the result instance of a
     constructor declaration. *)
  Example vcons_target :
    match lookup_symbol vec_ctx "vcons" with
    | Some t => constr_target t
    | None => None
    end = Some ("vec", [Var "s"; App (Cst "+1") (Var "n")]).
  Proof. reflexivity. Qed.

  (* NonEmptyVecNat: {vec nat (+1 n) :: |vcons}. *)
  Definition NonEmptyVec_sig : signature vec_ctx :=
    mk_signature vec_ctx ["vcons"] eq_refl.

  (* --- The positivity checker --- *)

  (* All example contexts pass. *)
  Example nat_ctx_wf : wf_symbol_ctx nat_ctx = true.
  Proof. reflexivity. Qed.
  Example list_ctx_wf : wf_symbol_ctx list_ctx = true.
  Proof. reflexivity. Qed.
  Example vec_ctx_wf : wf_symbol_ctx vec_ctx = true.
  Proof. reflexivity. Qed.

  (* succ : nat -> nat — the phantom (bare) recursive argument is
     rejected. *)
  Example phantom_succ_rejected :
    wf_symbol (mk_pi_type [("p", Cst "nat")] (Cst "nat")) = false.
  Proof. reflexivity. Qed.

  (* succ : {nat :: |0 |succ} -> {nat :: |0 |succ} — a signature target
     is not a constructor declaration at all. *)
  Example sig_target_rejected :
    wf_symbol (mk_pi_type [("p", Constr (Cst "nat") ["0"; "succ"])]
                 (Constr (Cst "nat") ["0"; "succ"])) = false.
  Proof. reflexivity. Qed.

  (* A negative occurrence — ({nat :: |0 |+1} -> nat) -> nat — is
     rejected: signatures may occur only positively in Δ_C. *)
  Example negative_occurrence_rejected :
    wf_symbol (mk_pi_type
                 [("f", Pi [("p", Constr (Cst "nat") ["0"; "+1"])]
                          (Cst "nat"))]
                 (Cst "nat")) = false.
  Proof. reflexivity. Qed.

  (* Context lookup: Gamma, ls : NonEmpty nat. *)
  Example lookup_ls :
    lookup_var (ctx_cons "ls" (Constr (App (Cst "list") (Cst "nat")) ["new"])
                  empty_ctx) "ls"
    = Some (Constr (App (Cst "list") (Cst "nat")) ["new"]).
  Proof. reflexivity. Qed.

  (* Substitution respects shadowing, including inside Pi telescopes. *)
  Example subst_shadow_lam :
    subst "x" (Cst "0") (Lam "x" (Var "x")) = Lam "x" (Var "x").
  Proof. reflexivity. Qed.

  Example subst_shadow_pi :
    subst "n" (Cst "0") (Pi [("n", Cst "nat")] (Var "n"))
    = Pi [("n", Cst "nat")] (Var "n").
  Proof. reflexivity. Qed.

  Example subst_pi :
    subst "A" (Cst "nat") (Pi [("x", Var "A")] (Var "A"))
    = Pi [("x", Cst "nat")] (Cst "nat").
  Proof. reflexivity. Qed.

  (* ξ-β: (λx. x) 0 --> 0. *)
  Example beta_step :
    App (Lam "x" (Var "x")) (Cst "0") --> Cst "0".
  Proof. exact (Xi_beta "x" (Var "x") (Cst "0") (value_cst "0")). Qed.

  (* ξ-case' then ξ-β:
     case (+1 0) of nat {0 => 0 | +1 => λp. p} -->* 0. *)
  Example pred_of_one :
    Match (App (Cst "+1") (Cst "0")) (Cst "nat")
      [("0", Cst "0"); ("+1", Lam "p" (Var "p"))]
    -->* Cst "0".
  Proof.
    eapply multi_step.
    { eapply (Xi_case' "+1" [Cst "0"]).
      - constructor; [apply value_cst | constructor].
      - reflexivity. }
    eapply multi_step.
    { exact (Xi_beta "p" (Var "p") (Cst "0") (value_cst "0")). }
    exact (multi_refl (Cst "0")).
  Qed.

  (* --- Typing the running examples --- *)

  (* Constant rule: 0 : nat. *)
  Example zero_nat : has_type nat_ctx empty_ctx (Cst "0") (Cst "nat").
  Proof. eapply T_Cst_eq; reflexivity. Qed.

  (* Application: list nat : Type. *)
  Example list_nat_sort :
    has_type list_ctx empty_ctx (App (Cst "list") (Cst "nat")) Sort.
  Proof.
    eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                     | eapply T_Cst_eq; reflexivity
                     | reflexivity | reflexivity | reflexivity].
  Qed.

  (* Rule 1: the signatures over list nat are types. *)
  Example list_sig_wf :
    has_type list_ctx empty_ctx
      (Constr (App (Cst "list") (Cst "nat")) ["new"; "empty"]) Sort.
  Proof. eapply T_Sig; [exact list_nat_sort | reflexivity | reflexivity]. Qed.

  Example nonempty_sig_wf :
    has_type list_ctx empty_ctx
      (Constr (App (Cst "list") (Cst "nat")) ["new"]) Sort.
  Proof. eapply T_Sig; [exact list_nat_sort | reflexivity | reflexivity]. Qed.

  (* Rule 3: empty nat : {list nat :: |new |empty}. *)
  Example empty_intro :
    has_type list_ctx empty_ctx (App (Cst "empty") (Cst "nat"))
      (Constr (App (Cst "list") (Cst "nat")) ["new"; "empty"]).
  Proof.
    assert (Hbase : has_type list_ctx empty_ctx
              (App (Cst "empty") (Cst "nat")) (App (Cst "list") (Cst "nat"))).
    { eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                       | eapply T_Cst_eq; reflexivity
                       | reflexivity | reflexivity | reflexivity]. }
    exact (T_SigIntro list_ctx empty_ctx _ _ "empty" [Cst "nat"]
             list_sig_wf eq_refl Hbase
             (inert_preservation list_ctx empty_ctx
                (App (Cst "empty") (Cst "nat"))
                (Constr (App (Cst "list") (Cst "nat")) ["new"; "empty"])
                eq_refl)).
  Qed.

  (* Rule 3 again: new nat 0 (empty nat) : {list nat :: |new}. *)
  Example new_intro :
    has_type list_ctx empty_ctx
      (App (App (App (Cst "new") (Cst "nat")) (Cst "0"))
           (App (Cst "empty") (Cst "nat")))
      (Constr (App (Cst "list") (Cst "nat")) ["new"]).
  Proof.
    assert (Hbase : has_type list_ctx empty_ctx
              (App (App (App (Cst "new") (Cst "nat")) (Cst "0"))
                   (App (Cst "empty") (Cst "nat")))
              (App (Cst "list") (Cst "nat"))).
    { eapply T_App_eq.
      - eapply T_App_eq.
        + eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                           | eapply T_Cst_eq; reflexivity
                           | reflexivity | reflexivity | reflexivity].
        + eapply T_Cst_eq; reflexivity.
        + reflexivity.
        + reflexivity.
        + reflexivity.
      - exact empty_intro.
      - reflexivity.
      - reflexivity.
      - reflexivity. }
    exact (T_SigIntro list_ctx empty_ctx _ _ "new"
             [Cst "nat"; Cst "0"; App (Cst "empty") (Cst "nat")]
             nonempty_sig_wf eq_refl Hbase
             (inert_preservation list_ctx empty_ctx
                (App (App (App (Cst "new") (Cst "nat")) (Cst "0"))
                   (App (Cst "empty") (Cst "nat")))
                (Constr (App (Cst "list") (Cst "nat")) ["new"])
                eq_refl)).
  Qed.

  (* Rule 4 + subsumption: a non-empty list is a list. *)
  Example new_intro_widened :
    has_type list_ctx empty_ctx
      (App (App (App (Cst "new") (Cst "nat")) (Cst "0"))
           (App (Cst "empty") (Cst "nat")))
      (Constr (App (Cst "list") (Cst "nat")) ["new"; "empty"]).
  Proof.
    eapply T_Sub; [exact new_intro | | reflexivity
                  | eapply inert_preservation; reflexivity].
    apply Sub_Sig; [apply conv_refl | reflexivity].
  Qed.

  (* Rule 2 + subsumption: forgetting the signature. *)
  Example new_intro_base :
    has_type list_ctx empty_ctx
      (App (App (App (Cst "new") (Cst "nat")) (Cst "0"))
           (App (Cst "empty") (Cst "nat")))
      (App (Cst "list") (Cst "nat")).
  Proof.
    eapply T_Sub; [exact new_intro | | reflexivity
                  | eapply inert_preservation; reflexivity].
    eapply Sub_SigBase; [exact list_nat_sort |].
    exists "list"; reflexivity.
  Qed.

  (* --- Rule 7 with pruning: the paper's head_nonempty example --- *)

  Definition vec_nat_1 : term :=
    apply_indexes (Cst "vec") [Cst "nat"; App (Cst "+1") (Cst "0")].

  (* Φ_ok at instance vec nat (+1 0): AGAINST refutes vempty (declared
     at index 0 against +1 0), so it is pruned. *)
  Example prune_vempty :
    prune_sig vec_ctx vec_nat_1 ["vempty"; "vcons"] = ["vcons"].
  Proof. reflexivity. Qed.

  (* With the positivity-compliant +1, its argument is typed at the
     signature: 0 : {nat :: |0 |+1} (Rule 3 with an empty spine). *)
  Example zero_sig_intro_vec :
    has_type vec_ctx empty_ctx (Cst "0") (Constr (Cst "nat") ["0"; "+1"]).
  Proof.
    assert (Hwf : has_type vec_ctx empty_ctx
              (Constr (Cst "nat") ["0"; "+1"]) Sort).
    { eapply T_Sig; [eapply T_Cst_eq; reflexivity
                    | reflexivity | reflexivity]. }
    assert (Hbase : has_type vec_ctx empty_ctx (Cst "0") (Cst "nat")).
    { eapply T_Cst_eq; reflexivity. }
    exact (T_SigIntro vec_ctx empty_ctx _ _ "0" [] Hwf eq_refl Hbase
             (inert_preservation vec_ctx empty_ctx (Cst "0")
                (Constr (Cst "nat") ["0"; "+1"]) eq_refl)).
  Qed.

  Example vec_nat_1_sort :
    has_type vec_ctx empty_ctx vec_nat_1 Sort.
  Proof.
    unfold vec_nat_1; cbn.
    eapply T_App_eq.
    - eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                       | eapply T_Cst_eq; reflexivity
                       | reflexivity | reflexivity | reflexivity].
    - eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                       | exact zero_sig_intro_vec
                       | reflexivity | reflexivity | reflexivity].
    - reflexivity.
    - reflexivity.
    - reflexivity.
  Qed.

  Example vec_sig_wf :
    has_type vec_ctx empty_ctx (Constr vec_nat_1 ["vempty"; "vcons"]) Sort.
  Proof.
    eapply T_Sig; [exact vec_nat_1_sort | reflexivity | reflexivity].
  Qed.

  Definition some_vec : term :=
    App (App (App (App (Cst "vcons") (Cst "nat")) (Cst "0"))
             (App (Cst "vempty") (Cst "nat")))
        (Cst "0").

  (* The recursive argument of vcons is supplied at signature type:
     vempty nat : {vec nat 0 :: |vempty |vcons}. *)
  Example vec_nat_0_sort :
    has_type vec_ctx empty_ctx
      (apply_indexes (Cst "vec") [Cst "nat"; Cst "0"]) Sort.
  Proof.
    cbn.
    eapply T_App_eq.
    - eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                       | eapply T_Cst_eq; reflexivity
                       | reflexivity | reflexivity | reflexivity].
    - eapply T_Cst_eq; reflexivity.
    - reflexivity.
    - reflexivity.
    - reflexivity.
  Qed.

  Example vempty_intro :
    has_type vec_ctx empty_ctx (App (Cst "vempty") (Cst "nat"))
      (Constr (apply_indexes (Cst "vec") [Cst "nat"; Cst "0"])
         ["vempty"; "vcons"]).
  Proof.
    assert (Hwf : has_type vec_ctx empty_ctx
              (Constr (apply_indexes (Cst "vec") [Cst "nat"; Cst "0"])
                 ["vempty"; "vcons"]) Sort).
    { eapply T_Sig; [exact vec_nat_0_sort | reflexivity | reflexivity]. }
    assert (Hbase : has_type vec_ctx empty_ctx
              (App (Cst "vempty") (Cst "nat"))
              (apply_indexes (Cst "vec") [Cst "nat"; Cst "0"])).
    { cbn.
      eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                       | eapply T_Cst_eq; reflexivity
                       | reflexivity | reflexivity | reflexivity]. }
    exact (T_SigIntro vec_ctx empty_ctx _ _ "vempty" [Cst "nat"]
             Hwf eq_refl Hbase
             (inert_preservation vec_ctx empty_ctx
                (App (Cst "vempty") (Cst "nat"))
                (Constr (apply_indexes (Cst "vec") [Cst "nat"; Cst "0"])
                   ["vempty"; "vcons"])
                eq_refl)).
  Qed.

  Example some_vec_ty :
    has_type vec_ctx empty_ctx some_vec (Constr vec_nat_1 ["vempty"; "vcons"]).
  Proof.
    assert (Hbase : has_type vec_ctx empty_ctx some_vec vec_nat_1).
    { unfold some_vec, vec_nat_1; cbn.
      eapply T_App_eq.
      - eapply T_App_eq.
        + eapply T_App_eq.
          * eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                             | eapply T_Cst_eq; reflexivity
                             | reflexivity | reflexivity | reflexivity].
          * eapply T_Cst_eq; reflexivity.
          * reflexivity.
          * reflexivity.
          * reflexivity.
        + exact vempty_intro.
        + reflexivity.
        + reflexivity.
        + reflexivity.
      - eapply T_Cst_eq; reflexivity.
      - reflexivity.
      - reflexivity.
      - reflexivity. }
    exact (T_SigIntro vec_ctx empty_ctx _ _ "vcons"
             [Cst "nat"; Cst "0"; App (Cst "vempty") (Cst "nat"); Cst "0"]
             vec_sig_wf eq_refl Hbase
             (inert_preservation vec_ctx empty_ctx some_vec
                (Constr vec_nat_1 ["vempty"; "vcons"]) eq_refl)).
  Qed.

  (* Eq_φ as the default conversion: the full signature converts to its
     pruned version, so some_vec is retyped at {vec nat (+1 0) ::
     |vcons} by Conversion_φ alone — no subtyping step involved. *)
  Example some_vec_pruned_ty :
    has_type vec_ctx empty_ctx some_vec (Constr vec_nat_1 ["vcons"]).
  Proof.
    eapply T_Conv.
    - exact some_vec_ty.
    - eapply T_Sig; [exact vec_nat_1_sort | reflexivity | reflexivity].
    - exact (conv_phi vec_ctx vec_nat_1 ["vempty"; "vcons"]).
    - reflexivity.
    - eapply inert_preservation; reflexivity.
  Qed.

  (* Rule 7: eliminating {vec nat (+1 0) :: |vempty |vcons} requires
     only the vcons branch, since Φ_ok = [vcons] — no vempty branch and
     no proof obligation. *)
  Example head_of_vec :
    has_type vec_ctx empty_ctx
      (Match some_vec (Cst "nat")
         [("vcons", Lam "s" (Lam "n" (Lam "xs" (Lam "x" (Cst "0")))))])
      (Cst "nat").
  Proof.
    eapply T_SigCase.
    - exact vec_sig_wf.
    - eapply T_Cst_eq; reflexivity.
    - exact some_vec_ty.
    - reflexivity.
    - reflexivity.
    - eapply bt_cons.
      + reflexivity.
      + cbn. apply T_Lam.
        * apply T_Sort.
        * cbn. apply T_Lam.
          -- eapply T_Cst_eq; reflexivity.
          -- cbn. apply T_Lam.
             ++ eapply T_Sig.
                ** eapply T_App_eq.
                   --- eapply T_App_eq; [eapply T_Cst_eq; reflexivity
                                        | apply T_Var; reflexivity
                                        | reflexivity | reflexivity
                                        | reflexivity].
                   --- apply T_Var; reflexivity.
                   --- reflexivity.
                   --- reflexivity.
                   --- reflexivity.
                ** reflexivity.
                ** reflexivity.
             ++ cbn. apply T_Lam.
                ** apply T_Var; reflexivity.
                ** cbn. eapply T_Cst_eq; reflexivity.
      + apply bt_nil.
    - intros v Hstep. inversion Hstep; subst.
      + exfalso.
        assert (Hinert : inertb some_vec = true)
          by (unfold some_vec; reflexivity).
        exact (inertb_irreducible some_vec Hinert e' H3).
      + unfold some_vec in H.
        change (apply_indexes (Cst c) us =
          apply_indexes (Cst "vcons")
            [Cst "nat"; Cst "0";
             App (Cst "vempty") (Cst "nat"); Cst "0"]) in H.
        destruct (apply_indexes_cst_inj c "vcons" us
          [Cst "nat"; Cst "0";
           App (Cst "vempty") (Cst "nat"); Cst "0"] H) as [-> ->].
        cbn in H4. inversion H4; subst N.
        eapply (T_MultiExpand vec_ctx empty_ctx _ (Cst "0") (Cst "nat")).
        2: { eapply T_Cst_eq; reflexivity. }
        cbn. eapply multi_step.
        { apply Xi_app1. apply Xi_app1. apply Xi_app1.
          apply Xi_beta. apply value_cst. }
        eapply multi_step.
        { apply Xi_app1. apply Xi_app1.
          apply Xi_beta. apply value_cst. }
        eapply multi_step.
        { apply Xi_app1. apply Xi_beta.
          apply value_cst_app1, value_cst. }
        eapply multi_step.
        { apply Xi_beta. apply value_cst. }
        apply multi_refl.
  Qed.

End Examples.
