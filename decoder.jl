# decoder.jl
using LinearAlgebra

# --- 1. THE DECODER LOGIC ---
"""
Compares a raw GPU vector against every known concept in Semantic Memory.
Returns the closest String symbol, or creates a new one if it's completely alien.
"""
function decode_vector_to_symbol(mem::SemanticMemory, raw_vector::Vector{Float32})
    # If memory is empty, this is the very first thing Prometheus has ever seen
    if mem.next_idx == 1
        new_symbol = "NOVEL_CONCEPT_1"
        get_or_create_concept!(mem, new_symbol, raw_vector)
        return new_symbol
    end
    
    best_similarity = -1.0
    best_match_idx = -1
    
    # Only compare against concepts we actually have
    active_count = mem.next_idx - 1
    
    # Extract the block of known embeddings
    known_embeddings = mem.concept_embeddings[:, 1:active_count]
    
    # Calculate Cosine Similarity against ALL known concepts simultaneously
    # Cosine Similarity = (A • B) / (||A|| * ||B||)
    norms = [norm(known_embeddings[:, i]) for i in 1:active_count]
    raw_norm = norm(raw_vector)
    
    for i in 1:active_count
        # Prevent division by zero if a dead mesh fired
        if norms[i] == 0.0 || raw_norm == 0.0
            continue
        end
        
        # LinearAlgebra dot() is highly optimized in Julia
        sim = dot(raw_vector, known_embeddings[:, i]) / (raw_norm * norms[i])
        
        if sim > best_similarity
            best_similarity = sim
            best_match_idx = i
        end
    end
    
    # --- THE NOVELTY THRESHOLD ---
    # If the vector is an 85% match or higher, we assume it's the same concept
    if best_similarity >= 0.85
        matched_symbol = mem.idx_to_concept[best_match_idx]
        return matched_symbol
    else
        # It's an alien feeling. Prometheus has never experienced this before.
        new_symbol = "NOVEL_CONCEPT_$(mem.next_idx)"
        get_or_create_concept!(mem, new_symbol, raw_vector)
        return new_symbol
    end
end