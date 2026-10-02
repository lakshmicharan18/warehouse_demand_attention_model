import numpy as np
import torch

from warehouse_demand_attention.data_generator import WarehouseDemandConfig, create_supervised_sequences, shifted_distribution_config
from warehouse_demand_attention.generalization import evaluate_frozen_model
from warehouse_demand_attention.training import WarehouseTrainingConfig, train_warehouse_model


def test_shifted_configuration_changes_only_intended_distribution_parameters():
    normal = WarehouseDemandConfig()
    shifted = shifted_distribution_config(normal)
    assert shifted.noise_std == normal.noise_std * 1.5
    assert shifted.spike_probability == normal.spike_probability * 2
    assert shifted.spike_magnitude_range[1] > normal.spike_magnitude_range[1]
    assert shifted.num_days == normal.num_days
    assert create_supervised_sequences(np.arange(30.0))[0].shape[1] == 24


def test_frozen_shifted_evaluation_keeps_parameters_and_normalizer():
    trained = train_warehouse_model(WarehouseTrainingConfig(embedding_dim=8, d_k=8, d_v=8, epochs=8, batch_size=128))
    before = [parameter.detach().clone() for parameter in trained.model.parameters()]
    normalizer = trained.data.normalizer
    result = evaluate_frozen_model(trained)
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(trained.model.parameters(), before))
    assert result.shifted_attention_predictions.shape == result.shifted_targets.shape
    assert np.isfinite(result.shifted_attention_predictions).all()
    assert result.shifted_config.noise_std > result.normal_config.noise_std
    assert normalizer is trained.data.normalizer
    assert set(result.shifted) == {"Last observation", "24-hour moving average", "Attention model"}
