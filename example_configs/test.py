config = {
    "_description": "TEST",
    "gpu": [0],
    
    # data
    "dataset": "Lsun_church",
    "data_path": "/root/notebooks/data/church_outdoor_train_lmdb/",
    "data_size": 2000,
    "use_image_generator": False,
    
    # model & training
    "model": "vanilla",
    "z_dim": 128,
    "gf_dim": 16,
    "df_dim": 16,
    "lr_g": 1e-4,
    "lr_d": 4e-4,
    "decay_rate": 0.95,
    "use_attention": True,
    "attn_dim_G": [16],
    "attn_dim_D": [4],
    "use_label": False,
    "batch_size_per_gpu": 64,
    "loss": "hinge_loss",
    "epoch": 10,
    "update_ratio": 1,
    
    # 
    "num_sample": 16,
    "summary_step_freq": 100,
    "output_path": "test"
}