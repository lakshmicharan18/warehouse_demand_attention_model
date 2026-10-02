import numpy as np

from warehouse_demand_attention.data_generator import (
    WarehouseDemandConfig,
    chronological_split,
    create_supervised_sequences,
    generate_warehouse_demand,
    shifted_distribution_config,
)


def test_same_seed_generates_identical_demand():
    first = generate_warehouse_demand(WarehouseDemandConfig(num_days=10, seed=42))
    second = generate_warehouse_demand(WarehouseDemandConfig(num_days=10, seed=42))
    assert np.array_equal(first.demand, second.demand)


def test_different_seeds_generate_different_demand():
    first = generate_warehouse_demand(WarehouseDemandConfig(num_days=10, seed=1))
    second = generate_warehouse_demand(WarehouseDemandConfig(num_days=10, seed=2))
    assert not np.array_equal(first.demand, second.demand)


def test_demand_is_non_negative_and_has_expected_hour_count():
    dataset = generate_warehouse_demand(WarehouseDemandConfig(num_days=7))
    assert len(dataset.demand) == 7 * 24
    assert np.all(dataset.demand >= 0)


def test_sequences_are_24_hours_with_immediate_next_hour_target():
    values = np.arange(30, dtype=float)
    X, y = create_supervised_sequences(values)
    assert X.shape == (6, 24)
    assert y.shape == (6,)
    assert np.array_equal(X[0], values[:24])
    assert y[0] == values[24]
    assert y[-1] == values[-1]


def test_chronological_splits_preserve_order_without_overlap():
    values = np.arange(100, dtype=float)
    X, y = create_supervised_sequences(values)
    splits = chronological_split(X, y)
    combined_targets = np.concatenate([splits.y_train, splits.y_validation, splits.y_test])
    assert np.array_equal(combined_targets, y)
    assert splits.y_train[-1] < splits.y_validation[0] < splits.y_test[0]
    assert len(splits.y_train) + len(splits.y_validation) + len(splits.y_test) == len(y)


def test_shifted_configuration_changes_relevant_parameters():
    normal = WarehouseDemandConfig()
    shifted = shifted_distribution_config(normal)
    assert shifted.noise_std > normal.noise_std
    assert shifted.spike_probability > normal.spike_probability
    assert shifted.spike_magnitude_range[1] > normal.spike_magnitude_range[1]
