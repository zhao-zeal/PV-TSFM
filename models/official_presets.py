"""Author ETTh1 scripts (or model-specific official PV/default recipes).

MMSP keeps the accepted 24 -> 24 windows and shared data protocols.
"""

PRESETS = {
    'DLinear': dict(seed=2021, batch_size=32, train_epochs=10, patience=3,
                    learning_rate=0.005, moving_avg=25, optimizer='adam',
                    lradj='type1', monitor='mse', drop_last=True),
    'PatchTST': dict(seed=2021, batch_size=128, train_epochs=100, patience=100,
                     learning_rate=0.0001, d_model=16, d_ff=128, e_layers=3,
                     n_heads=4, dropout=0.3, fc_dropout=0.3, head_dropout=0.,
                     optimizer='adam', lradj='type3', monitor='mse', drop_last=True),
    'FusionSF': dict(seed=42, batch_size=16, train_epochs=100, patience=20,
                     learning_rate=0.0016, optimizer='adamw_fusionsf',
                     lradj='fusion_cosine', monitor='mae', drop_last=False),
    'CrossUnet': dict(seed=2021, batch_size=128, train_epochs=100, patience=10,
                      learning_rate=0.0001, d_model=256, d_ff=512, e_layers=3,
                      n_heads=4, dropout=0.05, factor=10, optimizer='adam',
                      lradj='type3', monitor='mse', drop_last=True),
    'TimeXer': dict(seed=2021, batch_size=32, train_epochs=10, patience=3,
                    learning_rate=0.0001, d_model=512, d_ff=512, e_layers=2,
                    n_heads=8, dropout=0.1, factor=3, patch_len=12,
                    optimizer='adam', lradj='type1', monitor='mse', drop_last=False),
    'Chronos2': dict(seed=42, batch_size=32, monitor='mse'),
    'ChronosX': dict(seed=42, batch_size=32, eval_batch_size=8,
                     learning_rate=0.01, max_steps=5000, eval_steps=100,
                     gradient_accumulation_steps=2, optimizer='adamw_chronosx',
                     lradj='linear', monitor='token_ce', drop_last=False),
    'TimeLLM': dict(seed=2021, batch_size=24, eval_batch_size=8, train_epochs=100,
                    patience=10, learning_rate=0.01, d_model=32, d_ff=128,
                    e_layers=2, n_heads=8, dropout=0.1, factor=3, optimizer='adam',
                    lradj='type1_onecycle_init', monitor='mse', drop_last=True,
                    precision='bf16'),
    'TimeVLM': dict(seed=2024, batch_size=32, train_epochs=10, patience=5,
                    learning_rate=0.001, d_model=128, d_ff=768, e_layers=2,
                    n_heads=8, factor=3, dropout=0.1,
                    optimizer='adam', lradj='type1', monitor='mse',
                    drop_last=False, precision='fp16'),
}

LEGACY = {
    'DLinear': dict(seed=42, batch_size=64, train_epochs=50, patience=10,
                    learning_rate=0.0016, moving_avg=13),
    'PatchTST': dict(seed=42, batch_size=16, train_epochs=50, patience=20,
                     learning_rate=0.0002, d_model=512, d_ff=2048, e_layers=3,
                     n_heads=8, dropout=0.05),
}
