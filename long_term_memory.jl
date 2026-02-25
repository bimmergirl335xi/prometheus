# long_term_memory.jl
using Mmap
using Base.Threads

mutable struct DeepStorage
    # We map massive arrays directly to the SATA SSD array
    # This matrix might be 1,000,000 columns wide (experiences) by 1064 rows deep
    episodic_matrix::Matrix{Float32} 
    
    # A dictionary that lives in fast RAM to tell us WHERE in the massive file to look
    concept_index::Dict{String, Int}
end

function mount_sas_array(file_path::String, max_memories::Int)
    println("💾 Mounting Deep Storage from SAS Backplane...")
    
    # Calculate the file size needed (e.g., 1064 floats * 4 bytes * 1,000,000 = ~4.2 GB per file)
    dimensions = (1064, max_memories)
    
    # If the file doesn't exist, create it and pad it with zeros
    if !isfile(file_path)
        io = open(file_path, "w+")
        # Pre-allocate the massive file on the RAID array
        truncate(io, sizeof(Float32) * dimensions[1] * dimensions[2])
        close(io)
    end
    
    # THE MAGIC: Map the SATA file directly into Julia's virtual RAM space
    io = open(file_path, "r+")
    mmap_matrix = Mmap.mmap(io, Matrix{Float32}, dimensions)
    
    return DeepStorage(mmap_matrix, Dict{String, Int}())
end

# --- THE ASYNCHRONOUS RECALL ---
function deep_memory_recall(storage::DeepStorage, memory_index::Int)
    # Because pulling from SATA takes time, we run this asynchronously
    # so the rest of the brain can keep breathing and balancing.
    
    # Accessing mmap_matrix[..., memory_index] triggers the Linux OS to 
    # physically spin up the SAS backplane and read the SSDs.
    recalled_vector = storage.episodic_matrix[:, memory_index]
    
    return recalled_vector
end