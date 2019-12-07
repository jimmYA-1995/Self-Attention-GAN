config = {
    # data
    "dataset": "Lsun_church",
    "data_path": "/root/notebooks/data/Lsun_church_unlabeled_128",
    "data_size": -1,
    "use_image_generator": False,
    
    # model & training
    "model": "vanilla",
    "z_dim": 128,
    "gf_dim": 16,
    "df_dim": 16,
    "lr_g": 1e-4,
    "lr_d": 4e-4,
    "use_attention": True,
    "attn_dim": [32],
    "use_label": False,
    "batch_size": 64,
    "loss": "hinge_loss",
    "epoch": 100,
    "update_ratio": 1,
    
    # 
    "num_sample": 9,
    "summary_step_freq": 100,
    "output_path": "1122_church128_1attn-dwn_newSigma_z128_b64_e100"
}