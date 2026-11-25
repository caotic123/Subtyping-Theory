require open subtyping.encode;

symbol nat__sig_target : Set;
rule nat__sig_target ↪ natLabel;
symbol nat__sig_indexes : indexes_type;
rule nat__sig_indexes ↪ index_null;
symbol nat__sig_pattern_any : index_pattern_list;
rule nat__sig_pattern_any ↪ index_pattern_nil;
symbol nat__sig_delta : delta_tel;
rule nat__sig_delta ↪ mk_delta_tel nat__sig_indexes nat__sig_pattern_any;
symbol nat__sig_phi_all : constructor_list;
rule nat__sig_phi_all ↪ constructor_list_nil;

symbol vec__sig_target : Set;
rule vec__sig_target ↪ vecLabel;
symbol vec__sig_indexes : indexes_type;
rule vec__sig_indexes ↪ index_poly_succ (index_dependent_lift_succ natLabel (index_null));
symbol vec__sig_pattern_any : index_pattern_list;
rule vec__sig_pattern_any ↪ index_pattern_cons index_pat_var (index_pattern_cons index_pat_var (index_pattern_nil));
symbol vec__sig_delta : delta_tel;
rule vec__sig_delta ↪ mk_delta_tel vec__sig_indexes vec__sig_pattern_any;
symbol vec__sig_phi_all : constructor_list;
rule vec__sig_phi_all ↪ constructor_list_cons emptyLabel (constructor_list_cons consLabel (constructor_list_nil));

symbol id_sig : Π (v : nat), (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_var) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_cons emptyLabel (constructor_list_nil)))) → (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_var) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_cons emptyLabel (constructor_list_nil))));
rule id_sig ↪ λ (v : nat) (m : (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_var) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_cons emptyLabel (constructor_list_nil))))), m;

symbol head_nonempty : Π (v : nat), (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_term (index_term_ctor (+1 _0) (index_term_cons index_term_var index_term_nil))) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_nil))) → nat;
rule head_nonempty ↪ λ (v : nat) (m : (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_term (index_term_ctor (+1 _0) (index_term_cons index_term_var index_term_nil))) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_nil)))),
    (signature_case vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_term (index_term_ctor (+1 _0) (index_term_cons index_term_var index_term_nil))) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_nil)) natLabel m (case_branches_cons consLabel (constructor_list_nil) natLabel (λ (k : constructor_type consLabel), (+1 _0)) (case_branches_nil natLabel)));

symbol vec_to_nonempty_append : Π (v : nat), (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_var) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_cons emptyLabel (constructor_list_nil)))) → nat → (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_term (index_term_ctor (+1 _0) (index_term_cons index_term_var index_term_nil))) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_nil)));
rule vec_to_nonempty_append ↪ λ (v : nat) (m : (signature_term vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_var) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_cons emptyLabel (constructor_list_nil))))) (x : nat), (signature_intro vec__sig_target (mk_delta_tel vec__sig_indexes (index_pattern_cons (index_pat_var) (index_pattern_cons (index_pat_term (index_term_ctor (+1 _0) (index_term_cons index_term_var index_term_nil))) (index_pattern_nil)))) (constructor_list_cons consLabel (constructor_list_nil)) consLabel top cons);


