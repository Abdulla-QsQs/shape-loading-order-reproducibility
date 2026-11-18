#!/usr/bin/env python3
"""Matched-load-pool quasi-static elastoplastic sphere study.

Scientific question
-------------------
For identical finite pools of quasi-static loading events, does a stochastic
ordering policy conditioned on residual geometry produce a different response
than geometry-blind random ordering within a matched load-event pool?

This script executes one paired-study member.  Every policy consumes the same
load-event pool exactly once; only the ordering policy changes.  The mechanics
use incremental small-strain J2 plasticity with isotropic hardening on an
updated mesh.  Every event is loaded and unloaded before residual geometry is
carried forward.  The formulation is intentionally quasi-static: it contains
no mass, velocity, acceleration, contact duration, or stress-wave dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dolfin
import numpy as np
import pandas as pd
from dolfin import (
    ALE,
    BoxMesh,
    Cell,
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    Identity,
    LocalSolver,
    Measure,
    MeshFunction,
    Point,
    SpatialCoordinate,
    TensorFunctionSpace,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    XDMFFile,
    assemble,
    cells,
    dev,
    dot,
    dx,
    facets,
    grad,
    inner,
    parameters,
    solve,
    sqrt,
    sym,
    tr,
)


IMPACT_MARKER = 1
SUPPORT_MARKER = 2


@dataclass(frozen=True)
class LoadEvent:
    event_id: int
    direction_x: float
    direction_y: float
    direction_z: float
    peak_force: float

    @property
    def direction(self) -> np.ndarray:
        return np.array(
            [self.direction_x, self.direction_y, self.direction_z], dtype=float
        )


@dataclass
class SelectionResult:
    event_id: int
    probability: float
    normalized_entropy: float
    standardized_score: float


@dataclass
class GeometryState:
    center_of_mass: np.ndarray
    covariance: np.ndarray
    normalized_shape_change: np.ndarray
    principal_values: np.ndarray
    principal_axes: np.ndarray


@dataclass
class SolveRecord:
    target_fraction: float
    iterations: int
    residual_norm: float
    relative_residual: float
    converged: bool


def unit_vector(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        raise ValueError("A loading direction must have non-zero magnitude.")
    return vector / magnitude


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / len(values))
    return weights / total


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    if len(p) <= 1:
        return 0.0
    positive = p[p > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / math.log(len(p))


def generate_balanced_load_pool(
    event_count: int,
    pool_seed: int,
    peak_force: float,
) -> List[LoadEvent]:
    """Generate an antipodally balanced, sphere-wide direction pool.

    Antipodal pairing makes the vector sum of the complete pool zero to machine
    precision.  All policies therefore receive the same directional exposure.
    """

    if event_count < 2 or event_count % 2 != 0:
        raise ValueError("event_count must be an even integer of at least 2.")
    rng = np.random.default_rng(pool_seed)
    half = rng.normal(size=(event_count // 2, 3))
    half /= np.linalg.norm(half, axis=1)[:, None]
    directions = np.vstack([half, -half])
    order = rng.permutation(event_count)
    directions = directions[order]
    return [
        LoadEvent(
            event_id=index,
            direction_x=float(direction[0]),
            direction_y=float(direction[1]),
            direction_z=float(direction[2]),
            peak_force=float(peak_force),
        )
        for index, direction in enumerate(directions)
    ]


def select_fixed(
    remaining: Sequence[int],
    _pool: Sequence[LoadEvent],
    _state: GeometryState,
    _rng: np.random.Generator,
    _config: dict,
) -> SelectionResult:
    event_id = int(min(remaining))
    return SelectionResult(event_id, 1.0, 0.0, 0.0)


def select_random(
    remaining: Sequence[int],
    _pool: Sequence[LoadEvent],
    _state: GeometryState,
    rng: np.random.Generator,
    _config: dict,
) -> SelectionResult:
    candidates = np.asarray(sorted(remaining), dtype=int)
    probabilities = np.full(len(candidates), 1.0 / len(candidates))
    event_id = int(rng.choice(candidates, p=probabilities))
    return SelectionResult(
        event_id,
        float(1.0 / len(candidates)),
        normalized_entropy(probabilities),
        0.0,
    )


def feedback_probabilities(
    remaining: Sequence[int],
    pool: Sequence[LoadEvent],
    state: GeometryState,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(sorted(remaining), dtype=int)
    directions = np.vstack([pool[int(index)].direction for index in candidates])
    radius = float(config["geometry"]["radius"])
    selector = config["selection"]
    shape_weight = float(selector["shape_weight"])
    com_weight = float(selector["com_weight"])
    beta = float(selector["feedback_strength"])

    shape_scores = np.einsum(
        "ni,ij,nj->n", directions, state.normalized_shape_change, directions
    )
    com_scores = directions @ (state.center_of_mass / radius)
    raw_scores = shape_weight * shape_scores + com_weight * com_scores
    # Do not standardize away the magnitude of the geometric state.  With this
    # definition an undeformed sphere gives a uniform distribution, and the
    # feedback strengthens only as the dimensionless residual geometry grows.
    centered_scores = raw_scores - raw_scores.mean()
    probabilities = stable_softmax(beta * centered_scores)
    return candidates, probabilities, centered_scores


def select_feedback(
    remaining: Sequence[int],
    pool: Sequence[LoadEvent],
    state: GeometryState,
    rng: np.random.Generator,
    config: dict,
) -> SelectionResult:
    candidates, probabilities, scores = feedback_probabilities(
        remaining, pool, state, config
    )
    selected_position = int(rng.choice(np.arange(len(candidates)), p=probabilities))
    return SelectionResult(
        event_id=int(candidates[selected_position]),
        probability=float(probabilities[selected_position]),
        normalized_entropy=normalized_entropy(probabilities),
        standardized_score=float(scores[selected_position]),
    )


POLICIES = {
    "fixed": select_fixed,
    "random": select_random,
    "feedback": select_feedback,
}


def create_mapped_sphere(radius: float, resolution: int):
    mesh = BoxMesh(
        Point(-1.0, -1.0, -1.0),
        Point(1.0, 1.0, 1.0),
        int(resolution),
        int(resolution),
        int(resolution),
    )
    coordinates = mesh.coordinates()
    x = coordinates[:, 0].copy()
    y = coordinates[:, 1].copy()
    z = coordinates[:, 2].copy()
    x_sphere = x * np.sqrt(
        np.maximum(0.0, 1.0 - y**2 / 2.0 - z**2 / 2.0 + y**2 * z**2 / 3.0)
    )
    y_sphere = y * np.sqrt(
        np.maximum(0.0, 1.0 - z**2 / 2.0 - x**2 / 2.0 + z**2 * x**2 / 3.0)
    )
    z_sphere = z * np.sqrt(
        np.maximum(0.0, 1.0 - x**2 / 2.0 - y**2 / 2.0 + x**2 * y**2 / 3.0)
    )
    coordinates[:, 0] = radius * x_sphere
    coordinates[:, 1] = radius * y_sphere
    coordinates[:, 2] = radius * z_sphere
    mesh.init()
    return mesh


def cell_signed_determinants(mesh) -> np.ndarray:
    coordinates = mesh.coordinates()
    determinants = np.empty(mesh.num_cells(), dtype=float)
    for cell in cells(mesh):
        vertices = cell.entities(0)
        points = coordinates[vertices]
        matrix = np.column_stack(
            [points[1] - points[0], points[2] - points[0], points[3] - points[0]]
        )
        determinants[cell.index()] = float(np.linalg.det(matrix))
    return determinants


def mark_directional_caps(
    mesh,
    center: np.ndarray,
    direction: np.ndarray,
    area_fraction: float,
):
    """Mark approximately equal-area caps on opposing sides of the current mesh."""

    direction = unit_vector(direction)
    records = []
    total_area = 0.0
    for facet in facets(mesh):
        if not facet.exterior():
            continue
        midpoint = facet.midpoint()
        point = np.array([midpoint.x(), midpoint.y(), midpoint.z()], dtype=float)
        offset = point - center
        norm = float(np.linalg.norm(offset))
        score = float(np.dot(offset / norm, direction)) if norm > 0.0 else 0.0
        vertex_coordinates = mesh.coordinates()[facet.entities(0)]
        area = 0.5 * float(
            np.linalg.norm(
                np.cross(
                    vertex_coordinates[1] - vertex_coordinates[0],
                    vertex_coordinates[2] - vertex_coordinates[0],
                )
            )
        )
        total_area += area
        records.append((score, area, int(facet.index())))
    if not records or total_area <= 0.0:
        raise RuntimeError("The current mesh has no measurable exterior surface.")

    target_area = float(area_fraction) * total_area
    markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    impact_area = 0.0
    impact_indices = set()
    for _score, area, index in sorted(records, key=lambda item: item[0], reverse=True):
        markers[index] = IMPACT_MARKER
        impact_indices.add(index)
        impact_area += area
        if impact_area >= target_area:
            break

    support_area = 0.0
    for _score, area, index in sorted(records, key=lambda item: item[0]):
        if index in impact_indices:
            continue
        markers[index] = SUPPORT_MARKER
        support_area += area
        if support_area >= target_area:
            break

    if impact_area <= 0.0 or support_area <= 0.0:
        raise RuntimeError("Unable to construct non-zero opposing boundary caps.")
    return markers, impact_area, support_area, total_area


def local_project(expression, space, output: Optional[Function] = None) -> Function:
    if output is None:
        output = Function(space)
    trial = TrialFunction(space)
    test = TestFunction(space)
    measure = Measure("dx", domain=space.mesh())
    solver = LocalSolver(inner(trial, test) * measure, inner(expression, test) * measure)
    solver.factorize()
    solver.solve_local_rhs(output)
    return output


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0.0:
        return float("nan")
    threshold = percentile / 100.0 * cumulative[-1]
    position = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


class QuasiStaticSphereStudy:
    def __init__(
        self,
        config: dict,
        pool_seed: int,
        policy_name: str,
        policy_seed: int,
        run_id: str,
        output_directory: Path,
        write_xdmf: bool = False,
    ) -> None:
        if policy_name not in POLICIES:
            raise ValueError(f"Unknown policy '{policy_name}'.")
        if dolfin.MPI.size(dolfin.MPI.comm_world) != 1:
            raise RuntimeError("This audited implementation currently requires a serial run.")

        self.config = config
        self.pool_seed = int(pool_seed)
        self.policy_name = policy_name
        self.policy_seed = int(policy_seed)
        self.run_id = str(run_id)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(self.policy_seed)

        geometry = config["geometry"]
        material = config["material"]
        solver_cfg = config["solver"]
        study_cfg = config["study"]
        self.radius = float(geometry["radius"])
        self.mesh = create_mapped_sphere(
            self.radius, int(geometry["mesh_resolution"])
        )
        self.initial_cell_determinants = cell_signed_determinants(self.mesh)
        if np.any(np.abs(self.initial_cell_determinants) <= 1.0e-20):
            raise RuntimeError("The initial mesh contains a degenerate tetrahedron.")

        self.V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.W_tensor = TensorFunctionSpace(self.mesh, "DG", 0)
        self.W_scalar = FunctionSpace(self.mesh, "DG", 0)
        self.stress = Function(self.W_tensor, name="cauchy_stress")
        self.equivalent_plastic_strain = Function(
            self.W_scalar, name="equivalent_plastic_strain"
        )
        self.identity = Identity(3)
        self.dx = Measure("dx", domain=self.mesh)

        self.density = float(material["density"])
        self.young_modulus = float(material["young_modulus"])
        self.poisson_ratio = float(material["poisson_ratio"])
        self.initial_yield = float(material["yield_strength"])
        self.hardening_modulus = float(material["hardening_modulus"])
        self.mu = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
        self.lmbda = (
            self.young_modulus
            * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )
        self.bulk_modulus = self.lmbda + 2.0 * self.mu / 3.0

        self.absolute_tolerance = float(solver_cfg["absolute_tolerance"])
        self.relative_tolerance = float(solver_cfg["relative_tolerance"])
        self.maximum_iterations = int(solver_cfg["maximum_iterations"])
        self.relaxation_parameter = float(solver_cfg["relaxation_parameter"])
        self.linear_solver = str(solver_cfg["linear_solver"])
        self.area_fraction = float(geometry["cap_area_fraction"])
        self.probe_force = float(study_cfg["probe_force"])
        self.probe_every = int(study_cfg["probe_every"])
        self.load_schedule = [float(value) for value in study_cfg["load_schedule"]]
        if not self.load_schedule or self.load_schedule[-1] != 0.0:
            raise ValueError("load_schedule must end at zero to preserve load-unload cycles.")
        if max(self.load_schedule) != 1.0:
            raise ValueError("load_schedule must include the normalized peak value 1.0.")

        self.pool = generate_balanced_load_pool(
            int(study_cfg["events_per_pool"]),
            self.pool_seed,
            float(study_cfg["peak_force"]),
        )
        self.remaining = set(event.event_id for event in self.pool)
        self.reference_covariance = self.geometry_state().covariance.copy()
        self.initial_probe_stiffness = self.probe_stiffnesses()
        self.solve_records: List[dict] = []
        self.rows: List[dict] = []
        self.start_time = time.perf_counter()

        self.xdmf: Optional[XDMFFile] = None
        if write_xdmf:
            xdmf_path = self.output_directory / f"{self.run_id}_fields.xdmf"
            self.xdmf = XDMFFile(self.mesh.mpi_comm(), str(xdmf_path))
            self.xdmf.parameters["flush_output"] = True
            self.xdmf.parameters["functions_share_mesh"] = True

    def epsilon(self, displacement):
        return sym(grad(displacement))

    def elastic_stress_increment(self, strain_increment):
        return (
            self.lmbda * tr(strain_increment) * self.identity
            + 2.0 * self.mu * strain_increment
        )

    def return_mapping(self, displacement_increment):
        strain_increment = self.epsilon(displacement_increment)
        trial_stress = self.stress + self.elastic_stress_increment(strain_increment)
        trial_deviator = dev(trial_stress)
        equivalent_trial = sqrt(
            Constant(1.5) * inner(trial_deviator, trial_deviator)
            + Constant(dolfin.DOLFIN_EPS)
        )
        current_yield = (
            Constant(self.initial_yield)
            + Constant(self.hardening_modulus) * self.equivalent_plastic_strain
        )
        yield_function = equivalent_trial - current_yield
        yield_tolerance = Constant(1.0e-8 * self.initial_yield)
        plastic_condition = dolfin.gt(yield_function, yield_tolerance)
        positive_part = dolfin.conditional(
            plastic_condition, yield_function, Constant(0.0)
        )
        plastic_increment = positive_part / Constant(
            3.0 * self.mu + self.hardening_modulus
        )
        plastic_indicator = dolfin.conditional(
            plastic_condition,
            Constant(1.0),
            Constant(0.0),
        )
        flow_direction = (
            plastic_indicator
            * trial_deviator
            / (equivalent_trial + Constant(dolfin.DOLFIN_EPS))
        )
        beta = (
            Constant(3.0 * self.mu)
            * plastic_increment
            / (equivalent_trial + Constant(dolfin.DOLFIN_EPS))
        )
        updated_stress = trial_stress - beta * trial_deviator
        return updated_stress, flow_direction, beta, plastic_increment

    def algorithmic_stress_increment(
        self,
        strain_increment,
        flow_direction: Function,
        beta: Function,
    ):
        elastic = self.elastic_stress_increment(strain_increment)
        normal_correction = (
            Constant(3.0 * self.mu)
            * (
                Constant(3.0 * self.mu / (3.0 * self.mu + self.hardening_modulus))
                - beta
            )
            * inner(flow_direction, strain_increment)
            * flow_direction
        )
        deviatoric_correction = (
            Constant(2.0 * self.mu) * beta * dev(strain_increment)
        )
        return elastic - normal_correction - deviatoric_correction

    def geometry_state(self) -> GeometryState:
        coordinates = self.mesh.coordinates()
        center = self.center_of_mass()
        centered = coordinates - coordinates.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if hasattr(self, "reference_covariance"):
            reference = self.reference_covariance / np.trace(self.reference_covariance)
            normalized_change = covariance / np.trace(covariance) - reference
        else:
            normalized_change = np.zeros((3, 3), dtype=float)
        return GeometryState(
            center_of_mass=center,
            covariance=covariance,
            normalized_shape_change=normalized_change,
            principal_values=eigenvalues,
            principal_axes=eigenvectors,
        )

    def center_of_mass(self) -> np.ndarray:
        coordinate = SpatialCoordinate(self.mesh)
        volume = assemble(Constant(1.0) * self.dx)
        moments = [assemble(coordinate[i] * self.dx) for i in range(3)]
        return np.asarray([moment / volume for moment in moments], dtype=float)

    def mesh_quality(self) -> Dict[str, float]:
        current = cell_signed_determinants(self.mesh)
        determinant_ratios = current / self.initial_cell_determinants
        absolute_volumes = np.abs(current) / 6.0
        return {
            "minimum_signed_jacobian_ratio": float(determinant_ratios.min()),
            "minimum_cell_volume": float(absolute_volumes.min()),
            "maximum_cell_volume": float(absolute_volumes.max()),
            "inverted_cell_count": int(np.count_nonzero(determinant_ratios <= 0.0)),
        }

    def scalar_cell_values_and_volumes(self, field: Function) -> Tuple[np.ndarray, np.ndarray]:
        values = field.vector().get_local()
        dofmap = self.W_scalar.dofmap()
        cell_values = np.empty(self.mesh.num_cells(), dtype=float)
        volumes = np.empty(self.mesh.num_cells(), dtype=float)
        for cell in cells(self.mesh):
            dof = int(dofmap.cell_dofs(cell.index())[0])
            cell_values[cell.index()] = values[dof]
            volumes[cell.index()] = float(cell.volume())
        return cell_values, volumes

    def von_mises_field(self) -> Function:
        deviator = dev(self.stress)
        expression = sqrt(
            Constant(1.5) * inner(deviator, deviator)
            + Constant(dolfin.DOLFIN_EPS)
        )
        field = local_project(expression, self.W_scalar)
        field.rename("von_mises_stress", "cellwise_von_mises_stress")
        return field

    def probe_stiffness(self, direction: np.ndarray) -> float:
        state = self.geometry_state()
        markers, impact_area, _support_area, _total_area = mark_directional_caps(
            self.mesh,
            state.center_of_mass,
            unit_vector(direction),
            self.area_fraction,
        )
        ds = Measure("ds", domain=self.mesh, subdomain_data=markers)
        trial = TrialFunction(self.V)
        test = TestFunction(self.V)
        displacement = Function(self.V)
        direction = unit_vector(direction)
        pressure = self.probe_force / impact_area
        traction = Constant(tuple((-direction * pressure).tolist()))
        bilinear = inner(
            self.elastic_stress_increment(self.epsilon(trial)), self.epsilon(test)
        ) * self.dx
        linear = dot(traction, test) * ds(IMPACT_MARKER)
        boundary = DirichletBC(
            self.V, Constant((0.0, 0.0, 0.0)), markers, SUPPORT_MARKER
        )
        solve(
            bilinear == linear,
            displacement,
            boundary,
            solver_parameters={"linear_solver": self.linear_solver},
        )
        traction_direction = Constant(tuple((-direction).tolist()))
        average_displacement = assemble(
            dot(displacement, traction_direction) * ds(IMPACT_MARKER)
        ) / impact_area
        if abs(average_displacement) <= 1.0e-20:
            return float("inf")
        return float(self.probe_force / abs(average_displacement))

    def probe_stiffnesses(self) -> Dict[str, float]:
        return {
            "stiffness_x": self.probe_stiffness(np.array([1.0, 0.0, 0.0])),
            "stiffness_y": self.probe_stiffness(np.array([0.0, 1.0, 0.0])),
            "stiffness_z": self.probe_stiffness(np.array([0.0, 0.0, 1.0])),
        }

    def solve_load_target(
        self,
        target_fraction: float,
        event: LoadEvent,
        markers: MeshFunction,
        impact_area: float,
    ) -> Tuple[Function, SolveRecord]:
        """Advance from the previous converged load state to one target fraction."""

        direction = unit_vector(event.direction)
        ds = Measure("ds", domain=self.mesh, subdomain_data=markers)
        pressure = float(target_fraction) * float(event.peak_force) / impact_area
        traction = Constant(tuple((-direction * pressure).tolist()))
        boundary = DirichletBC(
            self.V, Constant((0.0, 0.0, 0.0)), markers, SUPPORT_MARKER
        )

        displacement_increment = Function(self.V, name="load_step_displacement")
        correction = Function(self.V)
        trial = TrialFunction(self.V)
        test = TestFunction(self.V)
        stress_iteration = Function(self.W_tensor)
        flow_iteration = Function(self.W_tensor)
        beta_iteration = Function(self.W_scalar)
        plastic_increment_iteration = Function(self.W_scalar)

        initial_residual: Optional[float] = None
        residual_norm = float("inf")
        relative_residual = float("inf")
        converged = False
        completed_iterations = 0

        for iteration in range(self.maximum_iterations + 1):
            (
                stress_expression,
                flow_expression,
                beta_expression,
                plastic_increment_expression,
            ) = self.return_mapping(displacement_increment)
            local_project(stress_expression, self.W_tensor, stress_iteration)
            local_project(flow_expression, self.W_tensor, flow_iteration)
            local_project(beta_expression, self.W_scalar, beta_iteration)
            local_project(
                plastic_increment_expression,
                self.W_scalar,
                plastic_increment_iteration,
            )

            residual_form = (
                inner(stress_iteration, self.epsilon(test)) * self.dx
                - dot(traction, test) * ds(IMPACT_MARKER)
            )
            residual_vector = assemble(residual_form)
            boundary.apply(residual_vector)
            residual_norm = float(residual_vector.norm("l2"))
            if initial_residual is None:
                initial_residual = max(residual_norm, 1.0e-30)
            relative_residual = residual_norm / initial_residual

            if (
                residual_norm <= self.absolute_tolerance
                or relative_residual <= self.relative_tolerance
            ):
                converged = True
                completed_iterations = iteration
                break
            if iteration >= self.maximum_iterations:
                completed_iterations = iteration
                break

            tangent_form = inner(
                self.algorithmic_stress_increment(
                    self.epsilon(trial), flow_iteration, beta_iteration
                ),
                self.epsilon(test),
            ) * self.dx
            tangent_matrix = assemble(tangent_form)
            right_hand_side = residual_vector.copy()
            right_hand_side *= -1.0
            boundary.apply(tangent_matrix, right_hand_side)
            correction.vector().zero()
            solve(
                tangent_matrix,
                correction.vector(),
                right_hand_side,
                self.linear_solver,
            )
            displacement_increment.vector().axpy(
                self.relaxation_parameter, correction.vector()
            )
            displacement_increment.vector().apply("insert")

        record = SolveRecord(
            target_fraction=float(target_fraction),
            iterations=int(completed_iterations),
            residual_norm=float(residual_norm),
            relative_residual=float(relative_residual),
            converged=bool(converged),
        )
        if not converged:
            raise RuntimeError(
                "Nonlinear solve failed at target fraction "
                f"{target_fraction:.6f}: residual={residual_norm:.6e}, "
                f"relative={relative_residual:.6e}."
            )

        # Re-evaluate at the converged increment before committing the state.
        (
            stress_expression,
            _flow_expression,
            _beta_expression,
            plastic_increment_expression,
        ) = self.return_mapping(displacement_increment)
        local_project(stress_expression, self.W_tensor, stress_iteration)
        updated_plastic_strain = Function(self.W_scalar)
        local_project(
            self.equivalent_plastic_strain + plastic_increment_expression,
            self.W_scalar,
            updated_plastic_strain,
        )
        self.stress.assign(stress_iteration)
        self.equivalent_plastic_strain.assign(updated_plastic_strain)
        return displacement_increment, record

    def shape_metrics(self, state: GeometryState) -> Dict[str, float]:
        coordinates = self.mesh.coordinates()
        extents = coordinates.max(axis=0) - coordinates.min(axis=0)
        principal_spreads = np.sqrt(np.maximum(state.principal_values, 0.0))
        spread_mean = float(principal_spreads.mean())
        shape_anisotropy = (
            float((principal_spreads.max() - principal_spreads.min()) / spread_mean)
            if spread_mean > 0.0
            else 0.0
        )
        volume = float(assemble(Constant(1.0) * self.dx))
        return {
            "extent_x": float(extents[0]),
            "extent_y": float(extents[1]),
            "extent_z": float(extents[2]),
            "volume": volume,
            "com_x": float(state.center_of_mass[0]),
            "com_y": float(state.center_of_mass[1]),
            "com_z": float(state.center_of_mass[2]),
            "com_magnitude": float(np.linalg.norm(state.center_of_mass)),
            "principal_spread_1": float(principal_spreads[0]),
            "principal_spread_2": float(principal_spreads[1]),
            "principal_spread_3": float(principal_spreads[2]),
            "shape_anisotropy": shape_anisotropy,
        }

    def material_metrics(self) -> Dict[str, float]:
        plastic_values, volumes = self.scalar_cell_values_and_volumes(
            self.equivalent_plastic_strain
        )
        total_volume = float(volumes.sum())
        mean_plastic = (
            float(np.sum(plastic_values * volumes) / total_volume)
            if total_volume > 0.0
            else 0.0
        )
        variance = (
            float(np.sum(volumes * (plastic_values - mean_plastic) ** 2) / total_volume)
            if total_volume > 0.0
            else 0.0
        )
        localization = (
            float(math.sqrt(max(variance, 0.0)) / mean_plastic)
            if mean_plastic > 1.0e-14
            else 0.0
        )
        von_mises = self.von_mises_field()
        stress_values, stress_volumes = self.scalar_cell_values_and_volumes(von_mises)
        stress_mean = float(
            np.sum(stress_values * stress_volumes) / stress_volumes.sum()
        )
        return {
            "plastic_strain_mean": mean_plastic,
            "plastic_strain_max": float(plastic_values.max()),
            "plastic_strain_p95": weighted_percentile(
                plastic_values, volumes, 95.0
            ),
            "plastic_strain_p99": weighted_percentile(
                plastic_values, volumes, 99.0
            ),
            "plastified_volume_fraction": float(
                volumes[plastic_values > 1.0e-12].sum() / total_volume
            ) if total_volume > 0.0 else 0.0,
            "plastic_localization_cv": localization,
            "von_mises_mean": stress_mean,
            "von_mises_max": float(stress_values.max()),
            "von_mises_p95": weighted_percentile(
                stress_values, stress_volumes, 95.0
            ),
        }

    @staticmethod
    def stiffness_anisotropy(stiffnesses: Dict[str, float]) -> float:
        values = np.array(
            [
                stiffnesses["stiffness_x"],
                stiffnesses["stiffness_y"],
                stiffnesses["stiffness_z"],
            ],
            dtype=float,
        )
        mean = float(values.mean())
        return float((values.max() - values.min()) / mean) if mean > 0.0 else 0.0

    def build_metrics_row(
        self,
        cycle: int,
        event: Optional[LoadEvent],
        selection: Optional[SelectionResult],
        cap_data: Optional[Tuple[float, float, float]],
        solve_records: Sequence[SolveRecord],
        force_probe: bool = False,
    ) -> dict:
        state = self.geometry_state()
        row = {
            "run_id": self.run_id,
            "pool_seed": self.pool_seed,
            "policy": self.policy_name,
            "policy_seed": self.policy_seed,
            "cycle": int(cycle),
            "event_id": int(event.event_id) if event is not None else -1,
            "direction_x": float(event.direction_x) if event is not None else np.nan,
            "direction_y": float(event.direction_y) if event is not None else np.nan,
            "direction_z": float(event.direction_z) if event is not None else np.nan,
            "peak_force": float(event.peak_force) if event is not None else 0.0,
            "selection_probability": (
                float(selection.probability) if selection is not None else np.nan
            ),
            "selection_entropy": (
                float(selection.normalized_entropy) if selection is not None else np.nan
            ),
            "selection_score": (
                float(selection.standardized_score) if selection is not None else np.nan
            ),
            "feedback_strength": float(
                self.config["selection"]["feedback_strength"]
            ),
            "impact_area": float(cap_data[0]) if cap_data is not None else np.nan,
            "support_area": float(cap_data[1]) if cap_data is not None else np.nan,
            "surface_area": float(cap_data[2]) if cap_data is not None else np.nan,
            "all_load_steps_converged": bool(
                all(record.converged for record in solve_records)
            ),
            "maximum_newton_iterations": int(
                max((record.iterations for record in solve_records), default=0)
            ),
            "maximum_residual_norm": float(
                max((record.residual_norm for record in solve_records), default=0.0)
            ),
            "elapsed_seconds": float(time.perf_counter() - self.start_time),
        }
        row.update(self.shape_metrics(state))
        row.update(self.material_metrics())
        row.update(self.mesh_quality())

        should_probe = force_probe or cycle == 0 or (
            self.probe_every > 0 and cycle % self.probe_every == 0
        )
        if should_probe:
            stiffnesses = self.probe_stiffnesses()
            row.update(stiffnesses)
            row["stiffness_anisotropy"] = self.stiffness_anisotropy(stiffnesses)
            row["normalized_stiffness_x"] = (
                stiffnesses["stiffness_x"] / self.initial_probe_stiffness["stiffness_x"]
            )
            row["normalized_stiffness_y"] = (
                stiffnesses["stiffness_y"] / self.initial_probe_stiffness["stiffness_y"]
            )
            row["normalized_stiffness_z"] = (
                stiffnesses["stiffness_z"] / self.initial_probe_stiffness["stiffness_z"]
            )
        else:
            for column in (
                "stiffness_x",
                "stiffness_y",
                "stiffness_z",
                "stiffness_anisotropy",
                "normalized_stiffness_x",
                "normalized_stiffness_y",
                "normalized_stiffness_z",
            ):
                row[column] = np.nan
        return row

    def write_partial_outputs(self, status: str, error: Optional[str] = None) -> None:
        metrics_path = self.output_directory / f"{self.run_id}_cycle_metrics.csv"
        diagnostics_path = (
            self.output_directory / f"{self.run_id}_solver_diagnostics.csv"
        )
        pool_path = self.output_directory / f"{self.run_id}_load_pool.csv"
        metadata_path = self.output_directory / f"{self.run_id}_metadata.json"
        pd.DataFrame(self.rows).to_csv(metrics_path, index=False)
        pd.DataFrame(self.solve_records).to_csv(diagnostics_path, index=False)
        pd.DataFrame([asdict(event) for event in self.pool]).to_csv(
            pool_path, index=False
        )
        config_text = json.dumps(self.config, sort_keys=True, separators=(",", ":"))
        import hashlib

        metadata = {
            "run_id": self.run_id,
            "status": status,
            "error": error,
            "pool_seed": self.pool_seed,
            "policy": self.policy_name,
            "policy_seed": self.policy_seed,
            "completed_cycles": max((int(row["cycle"]) for row in self.rows), default=0),
            "requested_cycles": len(self.pool),
            "configuration_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            "config": self.config,
            "derived_material": {
                "shear_modulus": self.mu,
                "lame_lambda": self.lmbda,
                "bulk_modulus": self.bulk_modulus,
            },
            "load_pool_vector_sum": np.sum(
                np.vstack([event.direction for event in self.pool]), axis=0
            ).tolist(),
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "dolfin": dolfin.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "mesh": {
                "vertices": int(self.mesh.num_vertices()),
                "cells": int(self.mesh.num_cells()),
            },
            "elapsed_seconds": float(time.perf_counter() - self.start_time),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def write_spatial_state(self, cycle: int, residual_displacement: Function) -> None:
        if self.xdmf is None:
            return
        residual_displacement.rename(
            "residual_displacement", "post_unloading_residual_displacement"
        )
        self.equivalent_plastic_strain.rename(
            "equivalent_plastic_strain", "accumulated_equivalent_plastic_strain"
        )
        von_mises = self.von_mises_field()
        self.xdmf.write(residual_displacement, float(cycle))
        self.xdmf.write(self.equivalent_plastic_strain, float(cycle))
        self.xdmf.write(von_mises, float(cycle))

    def run(self) -> pd.DataFrame:
        initial_row = self.build_metrics_row(
            cycle=0,
            event=None,
            selection=None,
            cap_data=None,
            solve_records=[],
            force_probe=True,
        )
        self.rows.append(initial_row)
        self.write_partial_outputs("running")

        selector = POLICIES[self.policy_name]
        minimum_allowed_ratio = float(
            self.config["solver"]["minimum_jacobian_ratio"]
        )
        try:
            for cycle in range(1, len(self.pool) + 1):
                state_before = self.geometry_state()
                selection = selector(
                    sorted(self.remaining),
                    self.pool,
                    state_before,
                    self.rng,
                    self.config,
                )
                event = self.pool[selection.event_id]
                self.remaining.remove(selection.event_id)
                markers, impact_area, support_area, total_area = mark_directional_caps(
                    self.mesh,
                    state_before.center_of_mass,
                    event.direction,
                    self.area_fraction,
                )

                residual_displacement = Function(
                    self.V, name="cycle_residual_displacement"
                )
                cycle_solve_records: List[SolveRecord] = []
                for step_index, target_fraction in enumerate(self.load_schedule, start=1):
                    increment, record = self.solve_load_target(
                        target_fraction,
                        event,
                        markers,
                        impact_area,
                    )
                    residual_displacement.vector().axpy(1.0, increment.vector())
                    residual_displacement.vector().apply("insert")
                    cycle_solve_records.append(record)
                    diagnostic = asdict(record)
                    diagnostic.update(
                        {
                            "run_id": self.run_id,
                            "pool_seed": self.pool_seed,
                            "policy": self.policy_name,
                            "cycle": cycle,
                            "event_id": event.event_id,
                            "step_index": step_index,
                        }
                    )
                    self.solve_records.append(diagnostic)

                ALE.move(self.mesh, residual_displacement)
                self.mesh.bounding_box_tree().build(self.mesh)
                quality = self.mesh_quality()
                if quality["inverted_cell_count"] > 0:
                    raise RuntimeError(
                        f"Mesh inversion detected after cycle {cycle}: "
                        f"{quality['inverted_cell_count']} cells."
                    )
                if quality["minimum_signed_jacobian_ratio"] < minimum_allowed_ratio:
                    raise RuntimeError(
                        f"Mesh Jacobian ratio {quality['minimum_signed_jacobian_ratio']:.6e} "
                        f"fell below the configured limit after cycle {cycle}."
                    )

                self.write_spatial_state(cycle, residual_displacement)
                row = self.build_metrics_row(
                    cycle=cycle,
                    event=event,
                    selection=selection,
                    cap_data=(impact_area, support_area, total_area),
                    solve_records=cycle_solve_records,
                    force_probe=(cycle == len(self.pool)),
                )
                self.rows.append(row)
                self.write_partial_outputs("running")
                print(
                    f"[{self.run_id}] cycle={cycle:03d}/{len(self.pool):03d} "
                    f"event={event.event_id:03d} policy={self.policy_name} "
                    f"A_g={row['shape_anisotropy']:.6e} "
                    f"p95={row['plastic_strain_p95']:.6e} "
                    f"Jmin={row['minimum_signed_jacobian_ratio']:.6e} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            self.write_partial_outputs("failed", error=repr(exc))
            raise
        finally:
            if self.xdmf is not None:
                self.xdmf.close()

        self.write_partial_outputs("complete")
        return pd.DataFrame(self.rows)


def validate_config(config: dict) -> None:
    required_sections = {"study", "geometry", "material", "selection", "solver"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError("Configuration is missing sections: " + ", ".join(sorted(missing)))
    if float(config["geometry"]["cap_area_fraction"]) <= 0.0:
        raise ValueError("cap_area_fraction must be positive.")
    if float(config["geometry"]["cap_area_fraction"]) >= 0.25:
        raise ValueError("cap_area_fraction must be less than 0.25 for opposing caps.")
    if int(config["study"]["events_per_pool"]) % 2 != 0:
        raise ValueError("events_per_pool must be even for antipodal balancing.")


def configure_fenics() -> None:
    dolfin.set_log_level(30)
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["allow_extrapolation"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool-seed", type=int, required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-xdmf", action="store_true")
    parser.add_argument("--events-override", type=int)
    parser.add_argument("--mesh-resolution-override", type=int)
    parser.add_argument("--peak-force-override", type=float)
    parser.add_argument("--feedback-strength-override", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_fenics()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.events_override is not None:
        config["study"]["events_per_pool"] = int(args.events_override)
    if args.mesh_resolution_override is not None:
        config["geometry"]["mesh_resolution"] = int(args.mesh_resolution_override)
    if args.peak_force_override is not None:
        config["study"]["peak_force"] = float(args.peak_force_override)
    if args.feedback_strength_override is not None:
        config["selection"]["feedback_strength"] = float(
            args.feedback_strength_override
        )
    validate_config(config)
    study = QuasiStaticSphereStudy(
        config=config,
        pool_seed=args.pool_seed,
        policy_name=args.policy,
        policy_seed=args.policy_seed,
        run_id=args.run_id,
        output_directory=Path(args.output_dir),
        write_xdmf=bool(args.write_xdmf),
    )
    study.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
