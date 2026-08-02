from common import BASE_CONFIG, run

config = BASE_CONFIG.copy()
config.update({
    "save_ckp_path": "./checkpoints/nonorm",
    "wandb_run_name": "experiment1.no_rmsnorm",
    "export_final": "./out/model/model_nonorm.pt",
    "use_rmsnorm": False,          # 关键：移除 RMSNorm
    # 可以调整学习率以稳定训练
    "lr_max": 3e-4,
    "lr_min": 3e-5,
})

if __name__ == "__main__":
    run(config)