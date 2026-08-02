from common import BASE_CONFIG, run

config = BASE_CONFIG.copy()
config.update({
    "save_ckp_path": "./checkpoints/nope",
    "wandb_run_name": "experiment3.no_rope",
    "export_final": "./out/model/model_nope.pt",
    "use_rope": False,             # 关键：不使用 RoPE
})

if __name__ == "__main__":
    run(config)