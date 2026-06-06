def calculate_gemma4_kv_exact():
    # Context window length
    N = 262144
    sliding_window = 1024
    
    # Model parameters from config.json
    num_sliding_layers = 25
    sliding_kv_heads = 8
    sliding_head_dim = 256
    
    num_global_layers = 5
    global_kv_heads = 2
    global_head_dim = 512
    
    # attention_k_eq_v is True: K and V share the same representation, so we store 1 tensor instead of 2.
    kv_factor = 1  
    
    # Precisions
    precisions = [
        ("16-bit (BF16/FP16)", 2),
        ("8-bit (INT8/FP8)", 1),
        ("4-bit (INT4)", 0.5)
    ]
    
    print("=== Model: Gemma 4 26B (huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated) ===")
    print(f"Context Window (N): {N:,} tokens")
    print(f"Sliding Window size (W): {sliding_window} tokens")
    print(f"Sliding layers: {num_sliding_layers} (Heads: {sliding_kv_heads}, Head Dim: {sliding_head_dim})")
    print(f"Global layers: {num_global_layers} (Heads: {global_kv_heads}, Head Dim: {global_head_dim})")
    print(f"K == V (attention_k_eq_v): True (halves the KV cache size!)\n")
    
    for prec_name, bytes_per_elem in precisions:
        # 1. KV cache for sliding layers:
        # Each sliding layer stores at most sliding_window (1024) tokens in KV Cache
        sliding_kv_bytes_per_layer = kv_factor * sliding_kv_heads * sliding_head_dim * sliding_window * bytes_per_elem
        total_sliding_kv_bytes = num_sliding_layers * sliding_kv_bytes_per_layer
        
        # 2. KV cache for global/full layers:
        # Each global layer stores the full context window N (262,144) tokens in KV Cache
        global_kv_bytes_per_layer = kv_factor * global_kv_heads * global_head_dim * N * bytes_per_elem
        total_global_kv_bytes = num_global_layers * global_kv_bytes_per_layer
        
        # Total KV Cache size
        total_kv_bytes = total_sliding_kv_bytes + total_global_kv_bytes
        total_kv_gib = total_kv_bytes / (2**30)
        total_kv_gb_dec = total_kv_bytes / (10**9)
        
        sliding_kv_gib = total_sliding_kv_bytes / (2**30)
        global_kv_gib = total_global_kv_bytes / (2**30)
        
        print(f"--- Precision: {prec_name} ---")
        print(f"  Sliding KV Cache: {sliding_kv_gib:.4f} GiB")
        print(f"  Global KV Cache:  {global_kv_gib:.4f} GiB")
        print(f"  Total KV Cache:   {total_kv_gib:.4f} GiB ({total_kv_gb_dec:.2f} GB)")
        
        # Check if it fits in 30GB VRAM (assuming 50GB model weights)
        model_weights_gb = 50.0
        total_vram_gb = 80.0
        remaining_vram_gb = total_vram_gb - model_weights_gb
        fits = total_kv_gib <= remaining_vram_gb
        fits_str = "YES" if fits else "NO"
        print(f"  Fits in 30 GB VRAM? {fits_str} (Remaining: {remaining_vram_gb - total_kv_gib:.2f} GiB)\n")

calculate_gemma4_kv_exact()
