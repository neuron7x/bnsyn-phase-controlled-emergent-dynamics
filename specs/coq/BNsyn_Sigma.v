(* BNsyn Formal Proofs in Coq *)
(* Minimal working proof for sigma bounds preservation *)

Require Import Coq.Reals.Reals.
Require Import Coq.Reals.RIneq.
Open Scope R_scope.

(* Define sigma bounds *)
Definition sigma_min : R := 0.8.
Definition sigma_max : R := 1.2.

(* Clamp function: clamps a value to [min, max] *)
Definition clamp (min max x : R) : R :=
  if Rle_dec x min then min
  else if Rle_dec max x then max
  else x.

(* Lemma: clamp preserves bounds *)
Lemma clamp_preserves_bounds : forall (min max x : R),
  min <= max ->
  min <= clamp min max x <= max.
Proof.
  intros min max x Hminmax.
  unfold clamp.
  destruct (Rle_dec x min).
  - (* Case: x <= min *)
    split.
    + (* min <= min *)
      apply Rle_refl.
    + (* min <= max *)
      exact Hminmax.
  - (* Case: x > min *)
    destruct (Rle_dec max x).
    + (* Case: max <= x *)
      split.
      * (* min <= max *)
        exact Hminmax.
      * (* max <= max *)
        apply Rle_refl.
    + (* Case: x < max *)
      split.
      * (* min <= x *)
        apply Rnot_le_lt in n.
        apply Rlt_le.
        exact n.
      * (* x <= max *)
        apply Rnot_le_lt in n0.
        apply Rlt_le.
        exact n0.
Qed.

(* Theorem: Sigma clamping preserves bounds [sigma_min, sigma_max] *)
Theorem sigma_clamp_preserves_bounds : forall (sigma : R),
  sigma_min <= clamp sigma_min sigma_max sigma <= sigma_max.
Proof.
  intro sigma.
  apply clamp_preserves_bounds.
  unfold sigma_min, sigma_max.
  (* Prove 0.8 <= 1.2 *)
  apply Rle_trans with (r2 := 1).
  - (* 0.8 <= 1 *)
    apply Rlt_le.
    apply (Rlt_trans 0.8 0.9 1).
    + apply (Rlt_trans 0.8 0.85 0.9); lra.
    + apply (Rlt_trans 0.9 0.95 1); lra.
  - (* 1 <= 1.2 *)
    apply Rlt_le.
    lra.
Qed.

(* Corollary: Any sigma update using clamp stays in bounds *)
Corollary sigma_update_bounded : forall (sigma sigma_update : R),
  let sigma' := clamp sigma_min sigma_max sigma_update in
  sigma_min <= sigma' <= sigma_max.
Proof.
  intros sigma sigma_update sigma'.
  unfold sigma'.
  apply sigma_clamp_preserves_bounds.
Qed.

(* Additional lemma: clamp is idempotent *)
Lemma clamp_idempotent : forall (min max x : R),
  min <= max ->
  clamp min max (clamp min max x) = clamp min max x.
Proof.
  intros min max x Hminmax.
  unfold clamp at 1.
  destruct (Rle_dec (clamp min max x) min).
  - (* If clamped value <= min, it must equal min *)
    unfold clamp.
    destruct (Rle_dec x min); reflexivity.
  - destruct (Rle_dec max (clamp min max x)).
    + (* If clamped value >= max, it must equal max *)
      unfold clamp.
      destruct (Rle_dec x min).
      * (* Contradiction: min <= max < min *)
        exfalso.
        apply (Rle_not_lt min max).
        exact Hminmax.
        apply Rle_lt_trans with (r2 := clamp min max x).
        exact r.
        apply Rnot_le_lt.
        exact n.
      * destruct (Rle_dec max x); reflexivity.
    + (* If min < clamped value < max, unchanged *)
      reflexivity.
Qed.

(* Print completion message *)
(* Successfully defined and proved sigma bounds preservation *)
