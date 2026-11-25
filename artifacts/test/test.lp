constant symbol buh : TYPE;
 symbol foo_ : buh;
symbol bar_: buh;

rule foo_ ↪ bar_;

assert ⊢ foo_ ≡ bar_;
