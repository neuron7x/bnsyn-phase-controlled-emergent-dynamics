from aoc.contracts import SigmaIndex


def test_sigma_formula_correctness() -> None:
    sigma = SigmaIndex(
        conflict_density=0.2,
        dispersion=0.4,
        revision_elasticity=0.6,
        convergence_slope=0.8,
    )
    expected = 0.2 * 0.35 + 0.4 * 0.25 + 0.6 * 0.25 + (1 - 0.8) * 0.15
    assert sigma.distance_to_transition == expected
