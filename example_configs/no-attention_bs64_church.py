config = {
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
    "lr_d": 4e-4,
    "use_attention": False,
    "use_label": False,
    "batch_size": 256,
    "loss": "hinge_loss",
    "epoch": 400,
    "update_ratio": 1,
    
    # 
    "num_sample": 16,
    "summary_step_freq": 100,
    "output_path": "no_attn_bs256_400epoch"
}