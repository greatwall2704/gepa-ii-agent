from unittest.mock import Mock, patch

import pytest

from gepa import optimize
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector


@patch("gepa.api.GEPAEngine.run")
def test_module_selector_string_round_robin(mock_run):
    """Test that module_selector='round_robin' works with optimize()."""
    mock_run.return_value = Mock(
        program_candidates=[{"test": "value"}],
        parent_program_for_candidate=[None],
        program_full_scores_val_set=[0.5],
        prog_candidate_val_subscores=[[]],
        program_at_pareto_front_valset=[set()],
        num_metric_calls_by_discovery=[1],
    )

    mock_adapter = Mock()
    mock_adapter.evaluate.return_value = Mock(outputs=[], scores=[])

    result = optimize(
        seed_candidate={"test": "value"},
        trainset=[],
        adapter=mock_adapter,
        reflection_lm=lambda x: "test response",
        module_selector="round_robin",
        max_metric_calls=1,
    )

    assert result is not None


@patch("gepa.api.GEPAEngine.run")
def test_module_selector_custom_instance(mock_run):
    """Test that module_selector accepts custom instances with optimize()."""
    mock_run.return_value = Mock(
        program_candidates=[{"test": "value"}],
        parent_program_for_candidate=[None],
        program_full_scores_val_set=[0.5],
        prog_candidate_val_subscores=[[]],
        program_at_pareto_front_valset=[set()],
        num_metric_calls_by_discovery=[1],
    )

    mock_adapter = Mock()
    mock_adapter.evaluate.return_value = Mock(outputs=[], scores=[])

    class CustomComponentSelector(ReflectionComponentSelector):
        def select_modules(self, state, trajectories, subsample_scores, candidate_idx, candidate):
            return ["test_component"]

    custom_selector = CustomComponentSelector()

    result = optimize(
        seed_candidate={"test": "value"},
        trainset=[],
        adapter=mock_adapter,
        reflection_lm=lambda x: "test response",
        module_selector=custom_selector,
        max_metric_calls=1,
    )

    assert result is not None


def test_module_selector_invalid_string_raises_error():
    """Test that invalid module_selector string raises AssertionError."""
    mock_adapter = Mock()
    mock_adapter.evaluate.return_value = Mock(outputs=[], scores=[])

    with pytest.raises(AssertionError, match="Unknown module_selector strategy"):
        optimize(
            seed_candidate={"test": "value"},
            trainset=[],
            adapter=mock_adapter,
            reflection_lm=lambda x: "test response",
            module_selector="invalid_strategy",
            max_metric_calls=1,
        )
