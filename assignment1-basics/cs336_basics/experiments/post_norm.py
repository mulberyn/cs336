from common import BASE_CONFIG, run

config = BASE_CONFIG.copy()
config.update({
    "save_ckp_path": "./checkpoints/post_norm",
    "wandb_run_name": "experiment2.post_norm",
    "export_final": "./out/model/model_post_norm.pt",
    "norm_position": "post",       # 关键：改为 post-norm
})

if __name__ == "__main__":
    run(config)