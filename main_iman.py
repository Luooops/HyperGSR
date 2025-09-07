import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F

# Add imangraphnet path to sys.path for imports
sys.path.append('src/models/imangraphnet')
from src.models.imangraphnet.model import Aligner, Generator, Discriminator
from src.models.imangraphnet.losses import *
from src.models.imangraphnet.config import *
from src.models.imangraphnet.load_data import load_data
from sklearn.model_selection import KFold
import time
import gc
from datetime import datetime

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB (aligned with train.py)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    else:
        return 0.0

def log_metrics(log_file, metrics):
    """Log metrics to file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {metrics}\n")

def prepare_graph_data(adj_matrix):
    """Convert adjacency matrix to PyTorch Geometric Data object"""
    # Convert numpy array to tensor if needed
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.from_numpy(adj_matrix).float()
    
    n = adj_matrix.shape[0]
    edge_index = torch.nonzero(adj_matrix).t()
    x = adj_matrix  # Use adjacency matrix as node features
    
    return Data(
        x=x, 
        pos_edge_index=edge_index,  # Changed from edge_index to pos_edge_index
        num_nodes=n, 
        adj=adj_matrix
    )

def adapt_to_iman_format(data_list):
    """Convert data to IMANGraphNet format"""
    adapted_data = []
    for data in data_list:
        # Create edge attributes from adjacency matrix
        edge_attr = data.adj[data.edge_index[0], data.edge_index[1]].unsqueeze(1)
        
        # Create new Data object with required attributes
        adapted_data.append(Data(
            x=data.adj,  # Use adjacency matrix as node features
            pos_edge_index=data.edge_index,  # Use existing edge indices
            edge_attr=edge_attr,  # Create edge attributes from adj matrix
            adj=data.adj  # Keep original adjacency matrix
        ))
    return adapted_data

def train_iman_graphnet(X_train_source, X_train_target, device, res_dir):
    """Train IMANGraphNet model (aligned with train.py format)"""
    # Initialize models
    aligner = Aligner().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize optimizers
    Aligner_optimizer = torch.optim.AdamW(aligner.parameters(), lr=0.025, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.025, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))
    
    # Record GPU memory and time (aligned with train.py)
    start_time = time.time()
    gpu_memory_usage = []
    peak_gpu_memory = 0.0
    
    # Print initial GPU info (aligned with train.py)
    if torch.cuda.is_available():
        print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.4f} MB")
    else:
        print("Running on CPU")
    
    # Store losses for plotting (aligned with train.py)
    train_al_losses = []
    train_ge_losses = []
    train_d_losses = []
    
    # Training loop
    for epoch in range(N_EPOCHS):
        aligner.train()
        generator.train()
        discriminator.train()
        
        epoch_al_loss = 0.0
        epoch_ge_loss = 0.0
        epoch_d_loss = 0.0
        
        Al_losses = []
        Ge_losses = []
        losses_discriminator = []
        
        for i in range(len(X_train_source)):
            # Record GPU memory usage (aligned with train.py)
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_memory_usage.append(current_gpu_memory)
                peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
            
            # Prepare data
            source_graph = X_train_source[i].to(device)
            target_graph = X_train_target[i].to(device)
            
            # Domain alignment
            A_output = aligner(source_graph)
            A_casted = A_output.view(N_SOURCE_NODES, N_SOURCE_NODES)
            
            # Create Data object for generator input
            gen_input = Data(
                x=A_casted,
                pos_edge_index=source_graph.pos_edge_index,
                edge_attr=source_graph.edge_attr,
                adj=A_casted
            ).to(device)
            
            # Generate target graph
            G_output = generator(gen_input)
            G_output_reshaped = G_output.view(1, N_TARGET_NODES, N_TARGET_NODES, 1)
            G_output_casted = prepare_graph_data(G_output_reshaped.squeeze().detach().cpu().numpy()).to(device)
            
            # Calculate losses
            target_matrix = target_graph.adj
            
            # Alignment loss
            kl_loss = Alignment_loss(target_matrix, A_output)
            Al_losses.append(kl_loss)
            
            # Generator loss
            Gg_loss = GT_loss(target_matrix, G_output)
            D_real = discriminator(target_graph)
            D_fake = discriminator(G_output_casted)
            G_adversarial = adversarial_loss(D_fake, torch.ones_like(D_fake, requires_grad=False))
            G_loss = G_adversarial + Gg_loss
            Ge_losses.append(G_loss)
            
            # Discriminator loss
            D_real_loss = adversarial_loss(D_real, torch.ones_like(D_real, requires_grad=False))
            D_fake_loss = adversarial_loss(D_fake.detach(), torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            losses_discriminator.append(D_loss)
            
            # Clear cache (aligned with train.py)
            torch.cuda.empty_cache()
            gc.collect()
        
        # Update models
        generator_optimizer.zero_grad()
        Ge_losses_mean = torch.mean(torch.stack(Ge_losses))
        Ge_losses_mean.backward(retain_graph=True)
        generator_optimizer.step()
        
        Aligner_optimizer.zero_grad()
        Al_losses_mean = torch.mean(torch.stack(Al_losses))
        Al_losses_mean.backward(retain_graph=True)
        Aligner_optimizer.step()
        
        discriminator_optimizer.zero_grad()
        losses_discriminator_mean = torch.mean(torch.stack(losses_discriminator))
        losses_discriminator_mean.backward(retain_graph=True)
        discriminator_optimizer.step()
        
        # Calculate epoch losses (aligned with train.py)
        epoch_al_loss = Al_losses_mean.item()
        epoch_ge_loss = Ge_losses_mean.item()
        epoch_d_loss = losses_discriminator_mean.item()
        
        # Store losses
        train_al_losses.append(epoch_al_loss)
        train_ge_losses.append(epoch_ge_loss)
        train_d_losses.append(epoch_d_loss)
        
        # Print progress (aligned with train.py format)
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Al Loss: {epoch_al_loss:.4f}, Ge Loss: {epoch_ge_loss:.4f}, D Loss: {epoch_d_loss:.4f}")
    
    # Calculate total training time (aligned with train.py)
    total_time = time.time() - start_time
    
    # Save losses as .npy files (aligned with train.py)
    np.save(f'{res_dir}/train_al_losses.npy', np.array(train_al_losses))
    np.save(f'{res_dir}/train_ge_losses.npy', np.array(train_ge_losses))
    np.save(f'{res_dir}/train_d_losses.npy', np.array(train_d_losses))
    
    # Save models as .pth files (aligned with train.py)
    aligner_path = f"{res_dir}/aligner.pth"
    generator_path = f"{res_dir}/generator.pth"
    discriminator_path = f"{res_dir}/discriminator.pth"
    torch.save(aligner.state_dict(), aligner_path)
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f"Models saved as {aligner_path}, {generator_path}, {discriminator_path}")
    
    # Save statistics (aligned with train.py format)
    stats_content = f"""Peak GPU Usage: {peak_gpu_memory:.4f} MB\nTraining Time: {total_time:.2f} seconds"""
    stats_file = f"{res_dir}/stats.txt"
    with open(stats_file, 'w') as f:
        f.write(stats_content)
    print(f"Statistics saved as {stats_file}")
    
    return aligner, generator

def main():
    # Set device (aligned with train.py)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load hr and lr data directly using your load_data function
    print("Loading HR and LR datasets...")
    source_data, target_data = load_data(
        src_path='gsr_data/lr_train_.csv',
        trg_path='gsr_data/hr_train_.csv', 
        node_size=160,
        target_node_size=268,
        feature_strategy="adj"
    )
    print(f"Loaded {len(source_data)} source samples and {len(target_data)} target samples")
    
    # Convert to numpy arrays for KFold splitting
    source_indices = np.arange(len(source_data))
    
    # KFold cross-validation with k=3
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold = 0
    
    for train_index, test_index in kf.split(source_indices):
        fold += 1
        print(f"\n=== Fold {fold}/3 ===")
        print(f"Train samples: {len(train_index)}, Test samples: {len(test_index)}")
        
        # Create results directory
        res_dir = f"results/imangraphnet/fold_{fold}"
        os.makedirs(res_dir, exist_ok=True)
        
        # Split data based on indices
        train_source = [source_data[i] for i in train_index]
        train_target = [target_data[i] for i in train_index]
        
        # Adapt data format for IMANGraphNet (your load_data already returns PyG Data objects)
        print("Adapting data format for IMANGraphNet...")
        train_source = adapt_to_iman_format(train_source)
        train_target = adapt_to_iman_format(train_target)
        
        # Train model
        print("Training IMANGraphNet...")
        aligner, generator = train_iman_graphnet(train_source, train_target, device, res_dir)
        
        # Clear memory (aligned with train.py)
        del aligner, generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nTraining complete! Results saved in results/imangraphnet/ directory.")

if __name__ == "__main__":
    main() 