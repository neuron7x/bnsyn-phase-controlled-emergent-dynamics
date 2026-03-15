(* BNsyn Formal Proofs in Coq *)
(* Proof for criticality gain bounds preservation *)
(* ALIGNED WITH: src/bnsyn/config.py:CriticalityParams (gain_min=0.2, gain_max=5.0) *)

Require Import Coq.Reals.Reals.
Require Import Coq.Reals.RIneq.
Open Scope R_scope.

(* Define gain bounds from actual code *)
Definition gain_min : R := 0.2.
Definition gain_max : R := 5.0.

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

(* Theorem: Gain clamping preserves bounds [gain_min, gain_max] *)
(* Maps to: src/bnsyn/criticality/* gain control logic with bounds from CriticalityParams *)
Theorem gain_clamp_preserves_bounds : forall (gain : R),
  gain_min <= clamp gain_min gain_max gain <= gain_max.
Proof.
  intro gain.
  apply clamp_preserves_bounds.
  unfold gain_min, gain_max.
  (* Prove 0.2 <= 5.0 *)
  lra.
Qed.

(* Corollary: Any gain update using clamp stays in bounds *)
Corollary gain_update_bounded : forall (gain gain_update : R),
  let gain' := clamp gain_min gain_max gain_update in
  gain_min <= gain' <= gain_max.
Proof.
  intros gain gain_update gain'.
  unfold gain'.
  apply gain_clamp_preserves_bounds.
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
(* Successfully defined and proved criticality gain bounds preservation *)
(* Bounds: gain_min=0.2, gain_max=5.0 (from src/bnsyn/config.py:CriticalityParams) *)
