# thalamus.jl
using LinearAlgebra
using LoopVectorization
using Base.Threads

# --- POSIX C-Standard Library Bindings ---
const libc = "libc.so.6"
const librt = "librt.so.1"
const libpthread = "libpthread.so.0"

mutable struct ThalamicGatekeeper
    # Number of incoming sensory streams (e.g., 9 Visual Lobes)
    num_streams::Int
    tensor_dim::Int
    
    # The actual POSIX pointers to the GPU output RAM
    sensory_ptrs::Vector{Ptr{Float32}}
    sensory_sems::Vector{Ptr{Cvoid}}
    
    # The gating weights (Modulated by top-down attention)
    attention_weights::Vector{Float32}
    
    # Pre-allocated zero-garbage buffers
    incoming_buffer::Matrix{Float32}
    fused_output::Vector{Float32}
    
    # Top-down attention matrix (Maps abstract thoughts to sensory filters)
    W_attention::Matrix{Float32}
end

function init_thalamus(num_lobes::Int, dim::Int)
    println("🧠 Initializing Thalamic Switchboard for $num_lobes Sensory Streams...")
    
    ptrs = Vector{Ptr{Float32}}(undef, num_lobes)
    sems = Vector{Ptr{Cvoid}}(undef, num_lobes)
    
    # 1. Connect to the 9 Visual Cortex V100s via POSIX Shared Memory
    for i in 1:num_lobes
        # Note: In C++ we named them starting from the actual GPU IDs, assuming 0-8 here for simplicity
        mem_name = "/prometheus_vis_out_$(i-1)"
        sem_name = "/prometheus_vis_lock_$(i-1)"
        
        O_RDONLY = 0o0000 
        shm_fd = ccall((:shm_open, librt), Cint, (Cstring, Cint, Cuint), mem_name, O_RDONLY, 0666)
        
        PROT_READ = 0x1
        MAP_SHARED = 0x01
        ptrs[i] = ccall((:mmap, libc), Ptr{Float32}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, off_t), 
                        C_NULL, dim * sizeof(Float32), PROT_READ, MAP_SHARED, shm_fd, 0)
                        
        sems[i] = ccall((:sem_open, libpthread), Ptr{Cvoid}, (Cstring, Cint), sem_name, 0)
    end
    
    return ThalamicGatekeeper(
        num_lobes, dim, ptrs, sems,
        ones(Float32, num_lobes), # Start with equal attention to all lobes
        zeros(Float32, dim, num_lobes),
        zeros(Float32, dim),
        randn(Float32, num_lobes, 256) .* 0.01f0 # 256 is the size of the Global Workspace active thought
    )
end

# --- THE FILTERING LOOP ---
function thalamic_routing_pass!(thalamus::ThalamicGatekeeper, workspace_thought::Vector{Float32}, global_sensory_bus::Channel{Vector{Float32}})
    
    # 1. TOP-DOWN ATTENTION (The Brain telling the Thalamus what to look for)
    # Multiply the current conscious thought by the attention matrix to generate bias
    attention_bias = thalamus.W_attention * workspace_thought[1:256]
    
    # Softmax to create a probability distribution of which GPU to listen to
    exp_bias = exp.(attention_bias)
    thalamus.attention_weights .= exp_bias ./ sum(exp_bias)

    # 2. RAPID HARDWARE INGESTION (Pulling from the 9 V100s)
    # We do this fast so we don't hold the locks and stall the GPUs
    for i in 1:thalamus.num_streams
        ccall((:sem_wait, libpthread), Cint, (Ptr{Cvoid},), thalamus.sensory_sems[i])
        
        # Copy from POSIX C-pointer directly into our pre-allocated Julia matrix column
        unsafe_copyto!(pointer(thalamus.incoming_buffer, (i-1)*thalamus.tensor_dim + 1), thalamus.sensory_ptrs[i], thalamus.tensor_dim)
        
        ccall((:sem_post, libpthread), Cint, (Ptr{Cvoid},), thalamus.sensory_sems[i])
    end
    
    # 3. THE GATING MECHANISM (Vectorized Matrix Multiplication)
    # We blend the 9 streams into a single 8192-D tensor based on the attention weights
    fill!(thalamus.fused_output, 0.0f0)
    
    for i in 1:thalamus.num_streams
        weight = thalamus.attention_weights[i]
        
        # If the brain is ignoring this stream (weight < 0.05), skip the math entirely (Sparsity!)
        if weight > 0.05f0
            @turbo for j in 1:thalamus.tensor_dim
                thalamus.fused_output[j] += thalamus.incoming_buffer[j, i] * weight
            end
        end
    end
    
    # 4. EVENT-DRIVEN ROUTING
    # Only wake up the Xeons if the fused output has significant energy (salience)
    salience = norm(thalamus.fused_output)
    
    if salience > 1.5f0 # Threshold can be modulated by the Limbic system later
        # Push the gated, focused thought onto the bus for the Global Workspace to process
        put!(global_sensory_bus, copy(thalamus.fused_output))
    end
end