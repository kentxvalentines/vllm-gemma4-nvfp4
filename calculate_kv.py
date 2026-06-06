def calculate_kv_cache(name, layers, kv_heads, head_dim, seq_len=262144, model_weights_gb=50.0, total_vram_gb=80.0):
    remaining_vram_gb = total_vram_gb - model_weights_gb
    
    print(f"=== Model: {name} ===")
    print(f"  Layers: {layers}, KV Heads: {kv_heads}, Head Dim: {head_dim}")
    print(f"  Context Window: {seq_len:,} tokens")
    print(f"  Available VRAM for KV Cache: {remaining_vram_gb:.2f} GB\n")
    
    precisions = [
        ("16-bit (BF16/FP16)", 2),
        ("8-bit (INT8/FP8)", 1),
        ("4-bit (INT4)", 0.5)
    ]
    
    print(f"  {'Precision':<25} | {'KV Cache Size (GB)':<20} | {'Fits in Remaining VRAM?':<25}")
    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*25}")
    
    for prec_name, bytes_per_elem in precisions:
        # KV Cache Size = 2 (for K and V) * layers * kv_heads * head_dim * seq_len * bytes_per_elem
        kv_cache_bytes = 2 * layers * kv_heads * head_dim * seq_len * bytes_per_elem
        kv_cache_gb_dec = kv_cache_bytes / (10**9) # Decimal GB (often used by GPU specs)
        kv_cache_gib = kv_cache_bytes / (2**30)   # Binary GiB (what PyTorch reports)
        
        fits = kv_cache_gib <= remaining_vram_gb
        fits_str = "YES" if fits else "NO"
        
        print(f"  {prec_name:<25} | {kv_cache_gib:6.2f} GiB ({kv_cache_gb_dec:6.2f} GB) | {fits_str:<25}")
    print("\n" + "="*50 + "\n")

# Typical configurations:
# 1. Gemma 2 27B (FP16 size ~54GB, let's assume it occupies 50GB if slightly quantized or pruned)
calculate_kv_cache("Gemma 2 27B", layers=46, kv_heads=16, head_dim=128)

# 2. Llama 3 70B / Qwen 2.5 72B (Quantized to 4-bit, weight size ~40GB - 50GB)
calculate_kv_cache("Llama 3 70B / Qwen 2.5 72B (4-bit)", layers=80, kv_heads=8, head_dim=128)

# 3. Mixtral 8x7B (Quantized to 8-bit or 4-bit, weight size ~30-50GB)
calculate_kv_cache("Mixtral 8x7B", layers=32, kv_heads=8, head_dim=128)

# 4. Command R+ (Quantized, e.g. 4-bit or 3-bit, weights ~50GB)
# Layers: 64, KV Heads: 8, Head Dim: 128 (hidden size 12288, query heads 96, GQA with 8 KV heads)
calculate_kv_cache("Command R+", layers=64, kv_heads=8, head_dim=128)
