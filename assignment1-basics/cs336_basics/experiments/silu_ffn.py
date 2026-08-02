from common import BASE_CONFIG, run

config = BASE_CONFIG.copy()
config.update({
    "save_ckp_path": "./checkpoints/silu_ffn",
    "wandb_run_name": "experiment4.silu_ffn",
    "export_final": "./out/model/model_silu_ffn.pt",
    "ffn_type": "silu",            # 关键：使用 SiLU（无门控）
    # 参数匹配：SiLU 通常用 d_ff = 4 * d_model
    "d_ff": 2048,                  # 4 * 512 = 2048
})

if __name__ == "__main__":
    run(config)