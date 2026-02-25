# brainstem_network.jl
using ZMQ
using Base.Threads

function start_brainstem_receiver(sensory_bus::Channel{Vector{Float32}})
    println("📡 [Brainstem] Thread $(threadid()) bound. Initializing Optic/Auditory Nerves...")

    # 1. Initialize the ZeroMQ Context and Socket
    ctx = ZMQ.Context()
    sensory_nerve = ZMQ.Socket(ctx, ZMQ.SUB)
    
    # Subscribe to absolutely everything coming over the wire (empty string means no filtering)
    ZMQ.subscribe(sensory_nerve, "")
    
    # 2. Synapse with the Edge Nodes
    # You will replace these IPs with the actual static IP addresses of your Pi 5s on your local network.
    left_hemisphere_ip = "192.168.1.101" 
    right_hemisphere_ip = "192.168.1.102"
    port = "5555"

    println("🔗 Attempting to synapse with Hemispheres at $left_hemisphere_ip and $right_hemisphere_ip...")
    
    # ZMQ brilliantly handles multiple connections on a single socket.
    # If a Pi 5 reboots or drops WiFi, ZMQ automatically reconnects in the background.
    ZMQ.connect(sensory_nerve, "tcp://$left_hemisphere_ip:$port")
    ZMQ.connect(sensory_nerve, "tcp://$right_hemisphere_ip:$port")

    println("✅ [Brainstem] Nerves connected. Awaiting reality tensors.")

    # 3. The Continuous Ingestion Loop
    while true
        # 1. Catch the raw payload (This blocks efficiently until data arrives)
        msg_bytes = ZMQ.recv(sensory_nerve)
        
        # 2. The Zero-Copy Reinterpretation
        # We don't loop through the bytes or parse them. We simply tell the Xeon CPU:
        # "Treat this block of memory as a Float32 array."
        reality_tensor = reinterpret(Float32, msg_bytes)
        
        # 3. Inject it into the White Matter
        # The put! function safely pushes the tensor onto the bus. 
        # If the CPU meshes get overwhelmed, the channel acts as natural backpressure.
        put!(sensory_bus, reality_tensor)
    end
end