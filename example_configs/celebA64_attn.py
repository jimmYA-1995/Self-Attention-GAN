config = {
    "description": "attn in the last 2 layer. double lr. after add ckpt mechanism",
    "gpu": [0],
    
    # data
    "dataset": "CelebA",
    "data_path": "/root/notebooks/data/celebA_64/",
    "data_size": -1,
    "use_image_generator": False,
    "img_size": 64,
    "num_classes": 1,
    
    # model & training
    "model": "vanilla",
    "z_dim": 128,
    "gf_dim": 16,
    "df_dim": 16,
    "lr_g": 2e-4,
    "lr_d": 8e-4,
    "decay_rate": 0.99,
    "use_attention": True,
    "attn_dim_G": [32, 64],
    "attn_dim_D": [8, 4],
    "use_label": False,
    "batch_size": 64,
    "loss": "hinge_loss",
    "epoch": 100,
    "update_ratio": 1,
    
    # 
    "num_sample": 16,
    "summary_step_freq": 100,
    "output_path": "1125_celebA64_2attn_lrx2_b64_epoch100"
}