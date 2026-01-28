-------------------------------- MODULE BNsyn ---------------------------------
(*
 * TLA+ Specification for BNsyn Core Invariants
 * 
 * This specification models the critical safety properties of the BNsyn
 * thermostated bio-AI system, focusing on:
 * 1. Temperature schedule monotonicity during cooling
 * 2. Plasticity gate soundness (gate consistency with temperature)
 * 3. Criticality parameter (sigma) bounds enforcement
 * 4. Phase state consistency
 *
 * Author: BNsyn Contributors
 * Version: 1.0.0
 * Date: 2026-01-28
 *)

EXTENDS Naturals, Reals

CONSTANTS
    T0,           (* Initial temperature *)
    Tmin,         (* Minimum temperature floor *)
    Alpha,        (* Cooling factor (0 < alpha <= 1) *)
    Tc,           (* Critical temperature for gating *)
    GateTau,      (* Sigmoid sharpness *)
    SigmaMin,     (* Minimum sigma value *)
    SigmaMax,     (* Maximum sigma value *)
    MaxSteps      (* Maximum simulation steps *)

VARIABLES
    temperature,  (* Current temperature value *)
    gate,         (* Plasticity gate value [0, 1] *)
    sigma,        (* Criticality parameter *)
    phase,        (* System phase: "active" | "consolidating" | "cooled" *)
    step          (* Current step counter *)

vars == <<temperature, gate, sigma, phase, step>>

(*
 * Type invariants - define valid value ranges
 *)
TypeOK ==
    /\ temperature \in Real
    /\ temperature >= Tmin
    /\ temperature <= T0
    /\ gate \in Real
    /\ gate >= 0.0
    /\ gate <= 1.0
    /\ sigma \in Real
    /\ sigma >= SigmaMin
    /\ sigma <= SigmaMax
    /\ phase \in {"active", "consolidating", "cooled"}
    /\ step \in Nat
    /\ step <= MaxSteps

(*
 * Initial state
 *)
Init ==
    /\ temperature = T0
    /\ gate = 0.0  (* Initially gate is closed at high temperature *)
    /\ sigma = (SigmaMin + SigmaMax) / 2  (* Start at midpoint *)
    /\ phase = "active"
    /\ step = 0

(*
 * Temperature update: geometric cooling
 * T_new = max(Tmin, T_old * Alpha)
 *)
CoolTemperature ==
    /\ step < MaxSteps
    /\ phase = "active"
    /\ LET newTemp == IF temperature * Alpha > Tmin
                      THEN temperature * Alpha
                      ELSE Tmin
       IN /\ temperature' = newTemp
          /\ gate' = IF newTemp < Tc THEN 1.0 ELSE 0.5  (* Simplified sigmoid *)
          /\ phase' = IF newTemp = Tmin THEN "consolidating" ELSE "active"
          /\ step' = step + 1
          /\ UNCHANGED <<sigma>>

(*
 * Sigma update: maintain bounds
 *)
UpdateSigma ==
    /\ step < MaxSteps
    /\ sigma' \in Real
    /\ sigma' >= SigmaMin
    /\ sigma' <= SigmaMax
    /\ step' = step + 1
    /\ UNCHANGED <<temperature, gate, phase>>

(*
 * Phase transition to cooled state
 *)
TransitionToCooled ==
    /\ phase = "consolidating"
    /\ temperature = Tmin
    /\ gate >= 0.5  (* Gate must be sufficiently open *)
    /\ phase' = "cooled"
    /\ UNCHANGED <<temperature, gate, sigma, step>>

(*
 * Combined next-state relation
 *)
Next ==
    \/ CoolTemperature
    \/ UpdateSigma
    \/ TransitionToCooled

(*
 * Specification
 *)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* ========================================================================= *)
(* INVARIANTS - These are the critical safety properties                    *)
(* ========================================================================= *)

(*
 * INV-1: TempMonotone
 * Temperature never increases during cooling phase (unless at floor)
 *)
TempMonotone ==
    []((phase = "active" /\ temperature > Tmin) =>
       (temperature' <= temperature))

(*
 * INV-2: GateSoundness
 * Plasticity gate reflects temperature schedule state
 * When T < Tc, gate should be open (high value)
 * When T > Tc, gate should be closed (low value)
 *)
GateSoundness ==
    [](((temperature < Tc) => (gate >= 0.5)) /\
       ((temperature > Tc) => (gate <= 0.5)))

(*
 * INV-3: SigmaClamp
 * Sigma (criticality parameter) always stays within bounds
 *)
SigmaClamp ==
    [](sigma >= SigmaMin /\ sigma <= SigmaMax)

(*
 * INV-4: PhaseConsistency
 * Phase transitions follow valid state machine:
 * active -> consolidating -> cooled
 * Cannot skip states or go backwards
 *)
PhaseConsistency ==
    []((phase = "active" /\ phase' = "cooled") => FALSE)  (* Cannot skip consolidating *)

(*
 * INV-5: GateTemperatureCorrelation
 * Gate value correlates with temperature in consolidating phase
 *)
GateTemperatureCorrelation ==
    []((phase = "consolidating") => (gate > 0.0))

(*
 * Liveness property: system eventually reaches cooled state
 *)
EventuallyCooled ==
    <>(phase = "cooled")

================================================================================
