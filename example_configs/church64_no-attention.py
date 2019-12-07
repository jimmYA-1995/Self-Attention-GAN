config = {
    "_description": "use SNGAN model.",
    
    # data
    "dataset": "Lsun_church",
    "data_path": "/root/notebooks/data/church_outdoor_train_lmdb/",
    "data_size": -1,
    "use_image_generator": False,
    
    # model & training
    "model": "vanilla",
    "z_dim": 128,
    "gf_dim": 16,
    "df_dim": 16,
    "lr_g": 1e-4,
    "lr_d": 2e-4,
    "decay_rate": 0.99,
    "use_attention": False,
    "attn_dim_G": [],
    "attn_dim_D": [],
    "use_label": False,
    "batch_size": 64,
    "loss": "hinge_loss",
    "epoch": 50,
    "update_ratio": 1,
    
    # 
    "num_sample": 16,
    "summary_step_freq": 100,
    "output_path": "1123_church64_SNGAN_z128_b64_e50"
}