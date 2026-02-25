# semantic_memory.jl
using LinearAlgebra

# --- 1. THE DATA STRUCTURE ---
mutable struct SemanticMemory
    # Dictionary for fast string lookups
    concept_to_idx::Dict{String, Int}
    
    # Array for fast index-to-string lookups
    idx_to_concept::Vector{String}
    
    # The Knowledge Graph: How concepts relate to each other (e.g., APPLE is related to RED)
    association_matrix::Matrix{Float32}
    
    # The Latent Dictionary: The 16-D GPU vector that defines each concept
    concept_embeddings::Matrix{Float32} 
    
    # Track how many concepts we currently know
    next_idx::Int
end

# --- 2. INITIALIZATION ---
function init_semantic_memory(max_concepts::Int = 100000)
    return SemanticMemory(
        Dict{String, Int}(),
        Vector{String}(undef, max_concepts),
        zeros(Float32, max_concepts, max_concepts),
        zeros(Float32, 16, max_concepts), # 16-D vectors from the GPU swarm
        1
    )
end

# --- 3. MEMORY OPERATIONS ---
function get_or_create_concept!(mem::SemanticMemory, concept::String, embedding::Vector{Float32})
    # If we already know it, just return the index
    if haskey(mem.concept_to_idx, concept)
        return mem.concept_to_idx[concept]
    end
    
    # Otherwise, register a new concept
    idx = mem.next_idx
    mem.concept_to_idx[concept] = idx
    mem.idx_to_concept[idx] = concept
    
    # Save its mathematical signature
    mem.concept_embeddings[:, idx] = embedding
    
    mem.next_idx += 1
    println("📚 [Memory] New concept registered: $concept (Index: $idx)")
    return idx
end