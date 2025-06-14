# Gray-Scott Latent Space Vectors

This tool extracts meaningful latent space vectors from Gray-Scott model GIF animations using a 3D convolutional autoencoder. These vectors capture the pattern dynamics in a compact representation, complementing the simulation parameters (f and k).

## What are Latent Space Vectors?

Latent space vectors are compressed, meaningful representations of the GIF movies. While the f and k parameters control the simulation, they don't fully capture the resulting patterns and dynamics. The autoencoder learns to encode the visual content itself into a compact representation (32 dimensions by default).

## Features

- Extracts 32-dimensional latent vectors representing the visual content of each GIF
- Combines content-based features with simulation parameters
- Visualizes the latent space using t-SNE dimensionality reduction
- Saves all data to CSV files for further analysis
- Integrates with existing clustering data

## How to Use

1. Make sure your Gray-Scott GIF files are in the `gif/` directory
2. Install required dependencies:
   ```
   pip install torch torchvision numpy pandas matplotlib pillow scikit-learn
   ```
3. Run the extraction script:
   ```
   ./extract_latent_vectors.py
   ```
4. The script will:
   - Load all GIF files and extract frames
   - Train a 3D convolutional autoencoder (or load an existing model)
   - Extract latent vectors for each GIF
   - Create visualizations of the latent space
   - Save all data to CSV files

## Output Files

- `gif/conv3d_autoencoder_grayscott.pth`: Trained autoencoder model
- `gif/latent_space_tsne.png`: t-SNE visualization of latent space (colored by f parameter)
- `gif/latent_space_tsne_k.png`: t-SNE visualization of latent space (colored by k parameter)
- `gif/grayscott_latent_vectors.csv`: CSV file with latent vectors for each GIF
- `gif/grayscott_combined_data.csv`: Combined data with clusters and latent vectors

## Configuration

You can modify these parameters at the top of the script:

- `LATENT_DIM`: Dimension of the latent space (default: 32)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `EPOCHS`: Number of training epochs (default: 20)
- `MAX_FRAMES`: Maximum number of frames to use from each GIF (default: 32)

## How It Works

1. **Data Loading**: GIF files are loaded and converted to sequence of grayscale frames
2. **Autoencoder**: A 3D convolutional autoencoder processes the sequences
3. **Encoding**: The encoder compresses the visual information into the latent space
4. **Dimensionality Reduction**: t-SNE is used to visualize the high-dimensional latent space
5. **Integration**: Latent vectors are combined with simulation parameters for comprehensive analysis

## Usage in Clustering

The latent vectors can be used for more meaningful clustering of Gray-Scott patterns, as they capture the visual content rather than just the input parameters. This allows for better understanding of the relationship between parameters and resulting patterns. 