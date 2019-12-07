config = {
    "_description": "downsampled attn layer in the last2 layer of models.",
    "gpu": [0],
    
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
    "lr_g": 2e-4,
    "lr_d": 7e-4,
    "decay_rate": 0.99,
    "use_attention": True,
    "attn_dim_G": [32, 64],
    "attn_dim_D": [8, 4],
    "use_label": False,
    "batch_size_per_gpu": 64,
    "loss": "hinge_loss",
    "epoch": 100,
    "update_ratio": 1,
    
    # 
    "num_sample": 16,
    "summary_step_freq": 100,
    "output_path": "1126_church64_2attn-dwn_lrx2_z128_b64_e100"
}