output_name = "test"
config = {
    "_description": "Test configuration",
    "gpu": [0],
    
    # data
    "dataset": "Lsun_church",
    "data_path": "/home/yct/data/church_outdoor_train_lmdb/Lsun_church_unlabeled_64",
    "data_size": 2000,
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
    "batch_size": 64,
    "loss": "hinge_loss",
    "epoch": 10,
    "update_ratio": 1,
    
    # 
    "num_sample": 16,
    "summary_step_freq": 100,
    "log_dir": "logs/{}".format(output_name),
    "ckpt_dir": "checkpoints/{}".format(output_name),
    "img_dir": "images/{}".format(output_name)
}