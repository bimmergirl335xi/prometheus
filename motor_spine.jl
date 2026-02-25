# motor_spine.jl
using ZMQ
using Base.Threads

# --- 1. THE BIOLOGICAL STATE ---
mutable struct MotorSpine
    # High-level intent from the Cortex (0.0 to 1.0)
    forward_drive::Float32 
    turn_drive::Float32
    
    # The Biological Metronome
    phase_clock::Float64 
end

function init_spinal_cord()
    return MotorSpine(0.0f0, 0.0f0, 0.0)
end

# --- 2. THE CENTRAL PATTERN GENERATOR (Walking Math) ---
# This translates a simple "Walk Forward" command into the exact 
# geometry for an 8-legged alternating tripod/tetrapod gait. 
function calculate_octapod_gait(spine::MotorSpine, delta_time::Float64)
    # If the brain is silent, the clock stops, and the legs return to neutral.
    if spine.forward_drive == 0.0f0 && spine.turn_drive == 0.0f0
        return zeros(Float32, 8) 
    end
    
    # The metronome speeds up based on how hard the brain wants to move
    spine.phase_clock += delta_time * (spine.forward_drive * 5.0) 
    
    leg_angles = zeros(Float32, 8)
    
    for i in 1:8
        # An alternating gait requires odds and evens to be 180 degrees (π) out of phase.
        phase_shift = (i % 2 == 0) ? pi : 0.0
        
        # The math of a physical step: y = Amplitude * sin(Time + Phase)
        # We factor in the turn_drive so legs on one side take longer strides if turning.
        turn_modifier = (i <= 4) ? spine.turn_drive : -spine.turn_drive
        amplitude = spine.forward_drive + turn_modifier
        
        leg_angles[i] = amplitude * sin(spine.phase_clock + phase_shift)
    end
    
    return leg_angles
end

# --- 3. THE NERVOUS SYSTEM LOOP ---
function run_spinal_cord(intent_bus::Channel{Vector{Float32}})
    println("🦴 [Motor Spine] Thread $(threadid()) bound. CPG Metronome running.")
    
    spine = init_spinal_cord()
    
    # 1. Setup the ZeroMQ Publisher (The Descending Nerve)
    ctx = ZMQ.Context()
    motor_nerve = ZMQ.Socket(ctx, ZMQ.PUB)
    # Bind to a different port than the sensory ingress
    ZMQ.bind(motor_nerve, "tcp://0.0.0.0:5556") 
    
    println("⚡ [Motor Spine] Broadcasting servo geometry on port 5556...")
    
    last_tick = time()
    
    # The physics loop runs at a strict, continuous interval
    while true
        current_time = time()
        dt = current_time - last_tick
        last_tick = current_time
        
        # 1. Listen for new intent from the Emergent Cortex
        # isready() ensures we don't freeze the physics loop waiting for the brain
        if isready(intent_bus)
            intent_vector = take!(intent_bus)
            spine.forward_drive = intent_vector[1]
            spine.turn_drive = intent_vector[2]
        end
        
        # 2. Generate the actual motor geometry
        # This returns an array of 8 exact angles for the servos
        motor_angles = calculate_octapod_gait(spine, dt)
        
        # 3. Fire to the Edge Hardware
        # We instantly serialize the 8 Float32s and blast them over WiFi
        # The Pi 5 / Arduino Giga will catch this instantly
        ZMQ.send(motor_nerve, reinterpret(UInt8, motor_angles))
        
        # Lock the spine to a 50Hz (20ms) physics loop. 
        # This gives the physical servos time to actually move to the target angle.
        sleep(0.02) 
    end
end