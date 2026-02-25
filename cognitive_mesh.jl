# cognitive_mesh.jl
using LinearAlgebra
using Base.Threads

# --- 1. THE GLOBAL WORKSPACE (Consciousness) ---
# This is the shared blackboard. It represents the single, unified 
# "thought" Prometheus is currently holding in its mind.
mutable struct GlobalWorkspace
    active_thought::Vector{Float32}
    confidence::Float32 # How sure the system is about this thought
    dominant_column_id::Int
end

# --- 2. THE HIERARCHICAL COLUMN ---
mutable struct HierarchicalColumn
    id::Int
    
    # Layer states (The abstractions getting deeper)
    L1_state::Vector{Float32} # e.g., 128-D (Features)
    L2_state::Vector{Float32} # e.g., 32-D  (Objects)
    L3_state::Vector{Float32} # e.g., 8-D   (Physics/Reasoning)
    
    # Bottom-Up Weights (Perception: converting raw data into abstractions)
    W_up_1::Matrix{Float32}
    W_up_2::Matrix{Float32}
    W_up_3::Matrix{Float32}
    
    # Top-Down Weights (Expectation: predicting the layer below)
    W_down_3::Matrix{Float32}
    W_down_2::Matrix{Float32}
    W_down_1::Matrix{Float32}
    
    # Connection to the Global Workspace
    W_workspace::Matrix{Float32}
    
    learning_rate::Float32
end

function init_deep_column(id::Int)
    # Dimensions: 1064 (Raw) -> 128 (L1) -> 32 (L2) -> 8 (L3)
    return HierarchicalColumn(
        id,
        zeros(Float32, 128), zeros(Float32, 32), zeros(Float32, 8),
        
        # Bottom-Up matrices (scaled to prevent math explosion)
        randn(Float32, 128, 1064) .* 0.01f0,
        randn(Float32, 32, 128) .* 0.01f0,
        randn(Float32, 8, 32) .* 0.01f0,
        
        # Top-Down matrices
        randn(Float32, 32, 8) .* 0.01f0,
        randn(Float32, 128, 32) .* 0.01f0,
        randn(Float32, 1064, 128) .* 0.01f0,
        
        # Reads the 32-D workspace and influences the L3 reasoning layer
        randn(Float32, 8, 32) .* 0.01f0, 
        
        0.001f0
    )
end

# --- 3. THE REASONING CYCLE ---
function process_deep_thought!(col::HierarchicalColumn, reality_tensor::Vector{Float32}, workspace::GlobalWorkspace)
    
    # === PHASE 1: TOP-DOWN EXPECTATION (Reasoning) ===
    # Layer 3 combines its own past state with the global workspace context
    col.L3_state = tanh.(col.W_workspace * workspace.active_thought)
    
    # The hierarchy predicts downwards
    predict_L2 = col.W_down_3 * col.L3_state
    predict_L1 = col.W_down_2 * predict_L2
    predict_reality = col.W_down_1 * predict_L1
    
    # === PHASE 2: BOTTOM-UP PERCEPTION (Experiencing) ===
    # Reality hits the sensors and flows upward
    actual_L1 = tanh.(col.W_up_1 * reality_tensor)
    actual_L2 = tanh.(col.W_up_2 * actual_L1)
    actual_L3 = tanh.(col.W_up_3 * actual_L2)
    
    # === PHASE 3: THE ERROR SIGNALS (Surprise) ===
    # How wrong was the reasoning at each level of abstraction?
    error_reality = reality_tensor - predict_reality
    error_L1 = actual_L1 - predict_L1
    error_L2 = actual_L2 - predict_L2
    
    total_surprise = norm(error_reality) + norm(error_L1) + norm(error_L2)
    
    # === PHASE 4: NEUROPLASTICITY (Learning the rules of reality) ===
    if total_surprise > 0.1f0
        # The column organically rewires its matrices to minimize these specific errors next time.
        # (Simplified gradient update for brevity)
        col.W_down_1 += col.learning_rate .* (error_reality * transpose(predict_L1))
        col.W_down_2 += col.learning_rate .* (error_L1 * transpose(predict_L2))
        
        # The abstract layers learn to build better representations
        col.W_up_1 += col.learning_rate .* (error_L1 * transpose(reality_tensor))
        col.W_up_2 += col.learning_rate .* (error_L2 * transpose(actual_L1))
    end
    
    # The column updates its internal states to reality for the next tick
    col.L1_state = actual_L1
    col.L2_state = actual_L2
    
    return total_surprise, actual_L2 # We use L2 (Objects/Concepts) to bid for the workspace
end

# --- 4. THE COGNITIVE ORCHESTRATOR ---
function run_agi_core(sensory_bus::Channel{Vector{Float32}})
    workspace = GlobalWorkspace(zeros(Float32, 32), 0.0f0, 0)
    columns = [init_deep_column(i) for i in 1:40] # 40 deep columns across the Xeons
    
    for reality_tensor in sensory_bus
        
        best_surprise = Inf32
        winning_thought = zeros(Float32, 32)
        winning_id = 0
        
        # 1. All columns attempt to reason about reality simultaneously
        Threads.@threads for i in 1:length(columns)
            col = columns[i]
            surprise, abstract_thought = process_deep_thought!(col, reality_tensor, workspace)
            
            # 2. THE COMPETITION FOR CONSCIOUSNESS
            # The column that is the LEAST surprised fundamentally "understands" 
            # the current moment better than the others.
            if surprise < best_surprise
                best_surprise = surprise
                winning_thought = abstract_thought
                winning_id = col.id
            end
        end
        
        # 3. UPDATING THE GLOBAL WORKSPACE
        # The winning column gets to overwrite the blackboard.
        # Now, on the NEXT tick, all 40 columns will use this thought as context!
        workspace.active_thought = winning_thought
        workspace.dominant_column_id = winning_id
        workspace.confidence = 1.0f0 / (1.0f0 + best_surprise)
        
        if workspace.confidence > 0.8f0
            # println("💡 Column $winning_id seized consciousness! (Confidence: $(round(workspace.confidence, digits=2)))")
            # Here, the workspace.active_thought can be routed to the Motor Spine
        end
    end
end