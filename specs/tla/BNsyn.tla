-------------------------------- MODULE BNsyn ---------------------------------
(*
 * TLA+ Specification for BNsyn Core Invariants
 * 
 * This specification models the critical safety properties of the BNsyn
 * thermostated bio-AI system, focusing on:
 * 1. Temperature schedule monotonicity during cooling
 * 2. Plasticity gate soundness (gate consistency with temperature)
 * 3. Criticality gain parameter bounds enforcement
 * 4. Phase state consistency
 *
 * Code Mapping:
 * - Temperature schedule: src/bnsyn/temperature/schedule.py
 * - Criticality control: src/bnsyn/criticality/* and src/bnsyn/config.py:CriticalityParams
 * - Temperature params: src/bnsyn/config.py:TemperatureParams
 *
 * Author: BNsyn Contributors
 * Version: 2.0.0 (corrected to match code semantics)
 * Date: 2026-01-28
 *)

EXTENDS Naturals, Reals

CONSTANTS
    T0,           (* Initial temperature *)
    Tmin,         (* Minimum temperature floor *)
    Alpha,        (* Cooling factor (0 < alpha <= 1) *)
    Tc,           (* Critical temperature for gating *)
    GateTau,      (* Sigmoid sharpness *)
    GainMin,      (* Minimum criticality gain *)
    GainMax,      (* Maximum criticality gain *)
    MaxSteps      (* Maximum simulation steps *)

VARIABLES
    temperature,  (* Current temperature value *)
    gate,         (* Plasticity gate value [0, 1] *)
    gain,         (* Criticality gain parameter *)
    phase,        (* System phase: "active" | "consolidating" | "cooled" *)
    step          (* Current step counter *)

vars == <<temperature, gate, gain, phase, step>>

(*
 * Type invariants - define valid value ranges
 * These correspond to code contracts in src/bnsyn/config.py
 *)
TypeOK ==
    /\ temperature \in Real
    /\ temperature >= Tmin
    /\ temperature <= T0
    /\ gate \in Real
    /\ gate >= 0.0
    /\ gate <= 1.0
    /\ gain \in Real
    /\ gain >= GainMin
    /\ gain <= GainMax
    /\ phase \in {"active", "consolidating", "cooled"}
    /\ step \in Nat
    /\ step <= MaxSteps

(*
 * Initial state
 *)
Init ==
    /\ temperature = T0
    /\ gate = 0.0  (* Initially gate is closed at high temperature *)
    /\ gain = (GainMin + GainMax) / 2  (* Start at midpoint *)
    /\ phase = "active"
    /\ step = 0

(*
 * Temperature update: geometric cooling
 * T_new = max(Tmin, T_old * Alpha)
 * Maps to: src/bnsyn/temperature/schedule.py:TemperatureSchedule.step()
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
          /\ UNCHANGED <<gain>>

(*
 * Gain update: maintain bounds via clamping
 * Maps to: src/bnsyn/criticality/* gain control with bounds from CriticalityParams
 *)
UpdateGain ==
    /\ step < MaxSteps
    /\ gain' \in Real
    /\ gain' >= GainMin
    /\ gain' <= GainMax
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
    /\ UNCHANGED <<temperature, gate, gain, step>>

(*
 * Combined next-state relation
 *)
Next ==
    \/ CoolTemperature
    \/ UpdateGain
    \/ TransitionToCooled

(*
 * Specification
 *)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* ========================================================================= *)
(* INVARIANTS - These are the critical safety properties                    *)
(* State predicates only - no primed variables                              *)
(* ========================================================================= *)

(*
 * INV-1: GainClamp
 * Criticality gain always stays within bounds
 * Maps to: src/bnsyn/config.py:CriticalityParams (gain_min=0.2, gain_max=5.0)
 *)
GainClamp ==
    gain >= GainMin /\ gain <= GainMax

(*
 * INV-2: TemperatureBounds
 * Temperature stays within physical bounds
 * Maps to: src/bnsyn/config.py:TemperatureParams (T0=1.0, Tmin=1e-3)
 *)
TemperatureBounds ==
    temperature >= Tmin /\ temperature <= T0

(*
 * INV-3: GateBounds
 * Plasticity gate stays in valid range [0, 1]
 * Maps to: src/bnsyn/temperature/schedule.py:gate_sigmoid return value
 *)
GateBounds ==
    gate >= 0.0 /\ gate <= 1.0

(*
 * INV-4: PhaseValid
 * Phase is always a valid state
 *)
PhaseValid ==
    phase \in {"active", "consolidating", "cooled"}

(* ========================================================================= *)
(* PROPERTIES - Temporal formulas for liveness and progress                 *)
(* ========================================================================= *)

(*
 * PROP-1: TemperatureMonotone
 * Temperature never increases during active cooling phase
 * Maps to: src/bnsyn/temperature/schedule.py geometric cooling T' = max(Tmin, T * alpha)
 *)
TemperatureMonotone ==
    []((phase = "active" /\ temperature > Tmin) =>
       [](temperature' <= temperature \/ phase' # "active"))

(*
 * PROP-2: EventuallyCooled
 * System eventually reaches cooled state (liveness)
 *)
EventuallyCooled ==
    <>(phase = "cooled")

(*
 * PROP-3: GateCorrelation
 * When temperature drops below Tc, gate should eventually open
 * Maps to: src/bnsyn/temperature/schedule.py:gate_sigmoid behavior
 *)
GateCorrelation ==
    []((temperature < Tc) => <>(gate > 0.5))

================================================================================
