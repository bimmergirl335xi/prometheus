# limbic_system.jl
using LinearAlgebra
using Base.Threads
using Dates

mutable struct EndocrineSystem
    # --- 1. THE CHEMICAL SCALARS (0.0 to 1.0) ---
    dopamine::Float32      # Curiosity / Consolidation
    noradrenaline::Float32 # Volatility / Attention spread
    serotonin::Float32     # Homeostasis / Fatigue
    cortisol::Float32      # Pain / Aversion
    
    # --- 2. TRAUMA & SHOCK MECHANICS ---
    is_in_shock::Bool
    shock_recovery_timer::Float32
    entropy_multiplier::Float32 # Forces noise into the Global Workspace during pain
    
    # --- 3. HARDWARE INTEROCEPTION (The Body) ---
    max_safe_g_force::Float32
    current_system_temp::Float32
    memory_pressure::Float32
end

function init_limbic_system()
    println("🩸 Initializing Endocrine System & Nociceptive Pathways...")
    return EndocrineSystem(
        0.1f0, 0.1f0, 0.5f0, 0.0f0,
        false, 0.0f0, 1.0f0,
        2.5f0, # Max safe G-force before triggering pain (e.g., 2.5G sudden impact)
        40.0f0, 0.0f0
    )
end

# --- 1. THE PAIN / SHOCK OVERRIDE ---
function trigger_nociceptive_shock!(limbic::EndocrineSystem, severity::Float32)
    println("⚠️ TRAUMA DETECTED: Systemic Nociceptive Override Triggered!")
    
    # 1. Flood the system with Cortisol
    limbic.cortisol = clamp(limbic.cortisol + severity, 0.0f0, 1.0f0)
    
    # 2. Induce "Shock" (Paralysis and Confusion)
    limbic.is_in_shock = true
    limbic.shock_recovery_timer = severity * 5.0f0 # E.g., a 1.0 severity fall causes a 5-second freeze
    
    # 3. Spike the entropy multiplier (This will be used to inject static into the Thalamus)
    limbic.entropy_multiplier = 1.0f0 + (severity * 10.0f0)
    
    # 4. Suppress Dopamine (You cannot learn complex language or logic while in intense pain)
    limbic.dopamine = 0.0f0
end

# --- 2. THE CHEMICAL ORCHESTRATOR ---
function process_endocrine_system!(
    limbic::EndocrineSystem, 
    dt::Float32, 
    global_surprise::Float32, 
    surprise_derivative::Float32, 
    accelerometer_data::Vector{Float32} # IMU data from the edge nodes
)
    
    # === PHASE 1: HARDWARE TRAUMA CHECK (The Spider Body) ===
    # Calculate absolute G-force from the IMU (X, Y, Z acceleration)
    g_force = norm(accelerometer_data) 
    
    if g_force > limbic.max_safe_g_force
        # Scale the pain relative to how hard the impact was
        trauma_severity = clamp((g_force - limbic.max_safe_g_force) / 5.0f0, 0.1f0, 1.0f0)
        trigger_nociceptive_shock!(limbic, trauma_severity)
    end
    
    # === PHASE 2: SHOCK RECOVERY ===
    if limbic.is_in_shock
        limbic.shock_recovery_timer -= dt
        if limbic.shock_recovery_timer <= 0.0f0
            limbic.is_in_shock = false
            limbic.entropy_multiplier = 1.0f0
            println("🩹 Shock subsided. Motor functions and cognition returning to baseline.")
        else
            # While in shock, Cortisol stays maxed, skipping normal emotional processing
            return 
        end
    end
    
    # === PHASE 3: METABOLIC DECAY (Chemicals wash out over time) ===
    limbic.dopamine *= 0.98f0
    limbic.noradrenaline *= 0.99f0
    limbic.cortisol *= 0.95f0 # Cortisol washes out relatively fast once the pain stops
    
    # === PHASE 4: INTRINSIC MOTIVATION CALCULATION ===
    
    # 1. Epistemic Drive (Dopamine)
    # Triggered when surprise is actively falling (the "Aha!" moment of understanding)
    if surprise_derivative < -0.1f0 
        limbic.dopamine = clamp(limbic.dopamine + abs(surprise_derivative), 0.0f0, 1.0f0)
    end
    
    # 2. Volatility Drive (Noradrenaline)
    # Triggered when the environment suddenly stops making sense
    if surprise_derivative > 0.5f0
        limbic.noradrenaline = clamp(limbic.noradrenaline + surprise_derivative, 0.0f0, 1.0f0)
    end
    
    # 3. Homeostatic Drive (Serotonin)
    # Rises smoothly as memory pressure and temperature increase, forcing the AI to "sleep" or idle
    limbic.serotonin = (limbic.memory_pressure * 0.7f0) + ((limbic.current_system_temp / 95.0f0) * 0.3f0)
end