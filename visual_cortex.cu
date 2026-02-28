// visual_cortex.cu
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

// --- 1. HARDWARE DISCOVERY ---
std::vector<int> discover_visual_lobes() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::vector<int> v100_ids;
    
    std::cout << "👁️ Interrogating PCIe Bus for Visual Cortex Hardware...\n";
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::string deviceName(prop.name);
        
        // Reserve 9 V100s for the Visual Cortex (Save 1 for Thalamus, ignore RTX 4000)
        if (deviceName.find("V100") != std::string::npos && v100_ids.size() < 9) {
            std::cout << "  ✅ [Occipital Lobe " << i << "] Assigned: " << prop.name << "\n";
            v100_ids.push_back(i);
        }
    }
    return v100_ids;
}

// --- 2. MEXICAN HAT TOPOLOGY KERNEL ---
__global__ 
void lateral_inhibition_kernel(int num_fields, int dim, float* states) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int field_id = blockIdx.y;

    if (row < dim && field_id < num_fields) {
        int idx = field_id * dim + row;
        
        float my_val = states[idx];
        float L1 = (field_id > 0) ? states[(field_id - 1) * dim + row] : 0.0f;
        float R1 = (field_id < num_fields - 1) ? states[(field_id + 1) * dim + row] : 0.0f;
        float L2 = (field_id > 1) ? states[(field_id - 2) * dim + row] : 0.0f;
        float R2 = (field_id < num_fields - 2) ? states[(field_id + 2) * dim + row] : 0.0f;
        
        float new_val = (my_val * 0.80f) + (L1 * 0.15f) + (R1 * 0.15f) - (L2 * 0.05f) - (R2 * 0.05f);
        states[idx] = new_val > 0.0f ? new_val : 0.0f; // ReLU
    }
}

// --- 3. LOCALIZED PREDICTIVE CODING KERNEL (Real-time Learning) ---
__global__ 
void local_hebbian_learning_kernel(int num_fields, int dim, float* weights, float* inputs, float* outputs, float learning_rate) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Target neuron
    int col = blockIdx.y; // Source neuron
    
    if (row < dim && col < dim) {
        // Simple Contrastive Hebbian Learning: 
        // If the input and output fire together, strengthen the connection.
        // We decay weights slightly to prevent infinite growth (Apoptosis).
        for(int field = 0; field < num_fields; ++field) {
            long long weight_idx = (long long)field * dim * dim + (long long)row * dim + col;
            float input_val = inputs[col];
            float output_val = outputs[field * dim + row];
            
            // Delta = (Pre * Post * Learning Rate) - (Weight Decay)
            float delta = (input_val * output_val * learning_rate) - (weights[weight_idx] * 0.0001f);
            weights[weight_idx] += delta;
        }
    }
}

// --- 4. THE VISUAL CORTEX MANAGER ---
void manage_occipital_lobe(int gpu_id, int num_fields, int field_dim) {
    cudaSetDevice(gpu_id);
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    long long total_weights = (long long)num_fields * field_dim * field_dim;
              
    // POSIX Shared Memory setup
    // 1. Ingest from Optic Nerve (Pi 5 network bridge handled elsewhere)
    int shm_in_fd = shm_open("/prometheus_optic_nerve", O_RDONLY, 0666);
    float* optic_nerve_ptr = (float*)mmap(0, field_dim * sizeof(float), PROT_READ, MAP_SHARED, shm_in_fd, 0);
    sem_t* optic_sem = sem_open("/prometheus_optic_lock", 0);

    // 2. Output to Thalamus (The Gatekeeper)
    std::string thal_mem_name = "/prometheus_vis_out_" + std::to_string(gpu_id);
    std::string thal_sem_name = "/prometheus_vis_lock_" + std::to_string(gpu_id);
    int shm_out_fd = shm_open(thal_mem_name.c_str(), O_CREAT | O_RDWR, 0666);
    ftruncate(shm_out_fd, field_dim * sizeof(float)); 
    float* thalamus_ptr = (float*)mmap(0, field_dim * sizeof(float), PROT_WRITE, MAP_SHARED, shm_out_fd, 0);
    sem_t* thalamus_sem = sem_open(thal_sem_name.c_str(), O_CREAT, 0666, 1);

    // Allocate VRAM
    float *d_inputs, *d_weights, *d_outputs;
    cudaMalloc(&d_inputs, field_dim * sizeof(float)); 
    cudaMalloc(&d_weights, total_weights * sizeof(float));
    cudaMalloc(&d_outputs, num_fields * field_dim * sizeof(float));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Base learning rate (This will eventually be modulated by the Limbic System)
    float dopamine_level = 0.001f; 

    while (true) {
        // 1. Pull raw visual data from the Optic Nerve
        sem_wait(optic_sem);
        cudaMemcpy(d_inputs, optic_nerve_ptr, field_dim * sizeof(float), cudaMemcpyHostToDevice);
        sem_post(optic_sem);
        
        // 2. cuBLAS Feedforward (Visual Perception)
        cublasSgemmStridedBatched(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            field_dim, 1, field_dim, &alpha,
            d_weights, field_dim, field_dim * field_dim, 
            d_inputs, field_dim, 0, &beta,
            d_outputs, field_dim, field_dim, num_fields
        );
        cudaDeviceSynchronize();
        
        // 3. 3D Lateral Competition (Mexican Hat)
        dim3 topo_threads(256, 1, 1);
        dim3 topo_blocks((field_dim + 255) / 256, num_fields, 1);
        lateral_inhibition_kernel<<<topo_blocks, topo_threads>>>(num_fields, field_dim, d_outputs);
        cudaDeviceSynchronize();
        
        // 4. Localized Structural Plasticity (Predictive Coding / Learning)
        // The mesh rewires itself instantly based on the visual input, no CPU needed.
        dim3 learn_threads(32, 32, 1);
        dim3 learn_blocks((field_dim + 31) / 32, (field_dim + 31) / 32, 1);
        local_hebbian_learning_kernel<<<learn_blocks, learn_threads>>>(num_fields, field_dim, d_weights, d_inputs, d_outputs, dopamine_level);
        cudaDeviceSynchronize();
        
        // 5. Push the dominant visual concept up to the Thalamus
        sem_wait(thalamus_sem); 
        cudaMemcpy(thalamus_ptr, d_outputs, field_dim * sizeof(float), cudaMemcpyDeviceToHost);
        sem_post(thalamus_sem); 
    }
}

// --- 5. VISUAL CORTEX BOOT SEQUENCE ---
int main() {
    std::cout << "👁️ PROMETHEUS VISUAL CORTEX IGNITING...\n";
    std::vector<int> visual_lobes = discover_visual_lobes();
    
    if (visual_lobes.empty()) {
        std::cerr << "FATAL: No GPUs available for Visual Cortex. Halting.\n";
        return -1;
    }

    std::vector<std::thread> lobe_threads;
    
    for (int gpu_id : visual_lobes) {
        // Calculate max fields dynamically, leaving 2.5GB overhead
        cudaSetDevice(gpu_id);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t usable_mem = free_mem - (2ULL * 1024 * 1024 * 1024 + 512ULL * 1024 * 1024); 
        int field_dim = 8192; 
        size_t mem_per_field = (size_t)field_dim * field_dim * sizeof(float);
        int dynamic_num_fields = usable_mem / mem_per_field;
        
        std::cout << "  ➡️ [Occipital Lobe " << gpu_id << "] Autonomic Scaling: " 
                  << dynamic_num_fields << " fields.\n";

        lobe_threads.push_back(std::thread(manage_occipital_lobe, gpu_id, dynamic_num_fields, field_dim));
    }
    
    for (auto& t : lobe_threads) t.join();
    return 0;
}