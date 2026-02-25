# reasoning.jl

# --- 1. TYPE DEFINITIONS (Replacing Python Enums/Dataclasses) ---
@enum ActionType SPEAK=1 RECURSIVE_THINK=2 RESEARCH=3 # Add your other action types

# Julia structs are grouped data, not classes with hidden state
struct Action
    action_type::ActionType
    payload::Any
    metadata::Dict{String, Any}
    emergent::Bool
end

# The Engine struct simply holds the references.
# We don't put functions "inside" it like Python classes.
struct ReasoningEngine
    memory::Any             # In a full build, type this as ::LongTermMemory
    abstraction_engine::Any # Type as ::AbstractionEngine
end


# --- 2. MULTIPLE DISPATCH FUNCTIONS (Replacing Class Methods) ---
# Notice how the function takes the engine as the first argument.

"""
Transitive Inference: Finds a logical path between two concepts via Breadth-First Search.
"""
function deduce_relation(engine::ReasoningEngine, concept_a::String, concept_b::String; max_depth::Int = 2)
    # Get matrix indices (Julia uses 1-based indexing, not 0-based!)
    start_node = get_or_create_idx(engine.memory.semantic, concept_a)
    end_node = get_or_create_idx(engine.memory.semantic, concept_b)
        
    if start_node == end_node
        return 1.0, [concept_a]
    end

    # Queue holds Tuples: (current_node, path_list, current_confidence)
    queue = Tuple{Int, Vector{String}, Float64}[(start_node, [concept_a], 1.0)]
    visited = Set{Int}()
    
    best_path = String[]
    best_score = 0.0

    while !isempty(queue)
        # popfirst! removes and returns the front of the array (like pop(0) in Python)
        current, path, score = popfirst!(queue)
        
        if length(path) > max_depth + 1
            continue
        end
        
        if current == end_node
            if score > best_score
                best_score = score
                best_path = path
            end
            continue
        end

        if current in visited
            continue
        end
        push!(visited, current)

        # Native Julia Matrix Access (Blisteringly fast, no NumPy needed)
        # We grab the specific row for the current concept
        row = engine.memory.semantic.association_matrix[current, :]
        
        # Native vector filtering: find all indices where the connection is > 0.2
        neighbors = findall(x -> x > 0.2, row)
        
        for neighbor in neighbors
            strength = Float64(row[neighbor])
            neighbor_name = engine.memory.semantic.idx_to_concept[neighbor]
            
            # Transitive property decay
            new_score = score * strength * 0.8
            
            # Create a new array for the path and push it to the queue
            new_path = copy(path)
            push!(new_path, neighbor_name)
            push!(queue, (neighbor, new_path, new_score))
        end
    end
    
    if best_score > 0.0
        println("🧠 [Reasoning] DeduceRelation: $concept_a -> $concept_b | Conf: $(round(best_score, digits=2))")
    end

    return best_score, best_path
end


"""
'Sanity Check': Breaks a sentence into Subject-Verb-Object to find contradictions.
"""
function validate_proposition(engine::ReasoningEngine, sentence::String)
    # Julia string manipulation is very similar to Python
    clean_sentence = uppercase(replace(sentence, "." => ""))
    
    if occursin(" IS ", clean_sentence)
        parts = split(clean_sentence, " IS ")
        
        # split() returns arrays. 'end' is a keyword for the last element in Julia.
        subj = split(strip(parts[1]))[end] 
        obj = split(strip(parts[2]))[1]
        
        # Check semantic memory for existing facts
        known_obj = query_fact(engine.memory.semantic, subj, "IS_A")
        
        if !isnothing(known_obj) && known_obj != obj
            similarity = get_similarity(engine.memory.semantic, known_obj, obj)
            if similarity < 0.3
                return false, "Conflict: I thought $subj was $known_obj, but text says $obj."
            end
        end
    end

    return true, "Proposition seems plausible."
end


"""
System 1 (Intuition) + System 2 (Logic).
"""
function hybrid_solve(engine::ReasoningEngine, goal_context::Dict{String, Any})
    
    # --- SYSTEM 1: INTUITION (Fast) ---
    candidates = String[]
    for key in keys(goal_context)
        associations = get_associated(engine.memory.semantic, key, top_k=5)
        # Fast array mapping syntax
        append!(candidates, [a[1] for a in associations]) 
    end
    
    if isempty(candidates)
        return nothing
    end
        
    # --- SYSTEM 2: LOGIC (Slow) ---
    best_action = nothing
    best_score = -1.0
    
    # unique() is the Julia equivalent of set() for arrays
    for concept in unique(candidates)
        try
            # Attempt to convert the string to the ActionType enum
            action_symbol = Symbol(uppercase(concept))
            action_type = eval(action_symbol) # Evaluates symbol to Enum
            
            # Check causal memory
            causal_score = predict_success(engine.memory.causal, action_type, goal_context)
            
            if causal_score > best_score
                best_score = causal_score
                
                metadata = Dict{String, Any}("source" => "hybrid_solve")
                best_action = Action(action_type, "Hybrid solution: $concept", metadata, false)
            end
        catch e
            # Ignore if the concept wasn't a valid ActionType Enum
            continue
        end
    end
            
    if !isnothing(best_action) && best_score > 0.3
        return best_action
    end
        
    return nothing
end