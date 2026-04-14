# Autoregressive Model for Prediction of Flow Around a Cylinder
## Problem Description

<div align="justify">
<div align="justify" style="text-align-last: left;">
</div>
The flow around a circular cylinder is a canonical benchmark problem in computational fluid dynamics (CFD), characterized by complex unsteady phenomena such as vortex shedding and wake formation. Accurately resolving these dynamics typically requires high-fidelity numerical simulations of the Navier–Stokes equations, which are computationally expensive, especially for long time horizons. In this work, we investigate a data-driven surrogate modeling approach for predicting the temporal evolution of the flow field using a convolutional neural network (CNN) based on the U-Net architecture. Despite recent advances in data-driven modeling, achieving accurate and stable long-term predictions across varying flow regimes and geometries remains a significant challenge.
</div>

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/1f203cd2-83df-42e3-8db6-744465508aa2" />
<div align="justify">
The U-Net is a convolutional neural network architecture originally developed for biomedical image segmentation, but it has proven exceptionally effective for fluid dynamics due to its ability to capture both local details and global context. Its main parts are: 
* Encoder-Decoder Structure: The encoder progressively downsamples the input, increasing the receptive field while reducing spatial resolution. Early encoder layers capture fine local features such as cylinder boundaries and small-scale velocity gradients, while deeper layers encode more abstract, global representations of the flow. The bottleneck, located at the middle of the “U”, contains the most compressed representation of the input, summarizing the overall flow structure such as wake formation and large-scale patterns. The decoder then progressively upsamples this representation to reconstruct the output field.
* Skip connections: The skip connections concatenate encoder and decoder feature maps at the same resolution level. Without skip connections the decoder only has the bottleneck to work from - spatial detail about the cylinder boundary, boundary layer, etc. is lost. With skip connections that detail is preserved and passed directly.

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/b578de99-5312-4f0d-8021-9e450250d988" />

The aim of this study is to develop deep learning models - specifically autoregressive models - for the prediction of velocity fields in flow around a cylinder across varying flow regimes and geometric configurations. The dataset incorporates variations in key physical and geometric parameters, including inlet velocity, fluid density, kinematic viscosity, as well as cylinder topology and radius.

The proposed models are trained to predict future flow states based on current observations. In particular, the two networks are designed to perform both one-step-ahead and direct multi-step (five-step-ahead) predictions of the velocity field, respectively. While the one-step-ahead model is trained to predict the immediate next state, it is often deployed in an autoregressive manner for long-term forecasting. In such cases, prediction errors may accumulate over successive time steps, potentially leading to degraded accuracy and instability. To mitigate this, a direct multi-step prediction model is also considered, which learns to predict future states without intermediate rollouts.

Formally, the learning task can be expressed as:

Input: (mask, Re, u(t), v(t)) ⟶ Output: (u(t+1), v(t+1)), and (mask, Re, u(t), v(t)) ⟶ (u(t+5), v(t+5)).

Here, the mask represents the spatial geometry of the domain, where a value of 0 denotes the presence of an obstacle (i.e., the cylinder), and 1 indicates the fluid region. The Reynolds number Re characterizes the flow regime, while 𝑢(𝑡) and 𝑣(𝑡) correspond to the x- and y-components of the velocity field at time 𝑡, respectively. The dataset used in this study is derived from CFDBench, a benchmark suite designed to evaluate the generalization capabilities of neural operators in computational fluid dynamics tasks.

## Methodology
The dataset is divided into three subsets based on the varying physical parameters: boundary conditions ("bc"), geometric properties ("geo"), and fluid properties ("prop"). Each subset differs in both the number of samples and the type of variation introduced. A summary of the dataset composition is provided in the table below. The input data are provided as .json files containing simulation parameters, while the output data are stored as .npy files representing velocity fields on a 64×64 grid across multiple timesteps.

| Subset | Description | Original Samples | Selected Samples | Timesteps per Sample
| :---: | :---: | :---: | :---: | :---: |
| **bc** | Inlet Velocity Variations | 50 | 20 | 620
| **geo** | Cylinder Topology & Radius | 20 | 19 | 2000
| **prop** | Density & Viscosity Variations | 116 | 73 | 1000

The original dataset exhibited an imbalance, as only a limited number of cases corresponded to Reynolds numbers greater than 400 and density greater than 20. Such imbalance can lead to biased model behavior and reduced predictive performance, particularly in underrepresented flow regimes. To address this issue, a subset of the original samples was excluded during dataset construction, resulting in a reduced but more balanced dataset. Furthermore, the dataset was temporally subsampled by selecting snapshots at intervals of five timesteps (e.g., 500, 505, 510, etc.), in order to limit the total number of samples and maintain a manageable computational cost during training. Finally, the initial timesteps of each simulation were omitted to exclude transient effects and focus on the more stable flow dynamics. The dataset was split into proportions of 0.7, 0.15, and 0.15 for the training, validation, and test sets, respectively. A stratified sampling strategy was employed to ensure that each split contains a balanced representation of all three subsets (“bc”, “geo”, and “prop”). Additionally, normalization was applied to the data to ensure consistent scaling and improve training stability. The normalization statistics were computed using only the training set in order to avoid data leakage and ensure a fair evaluation. Finally, a weighted sampling strategy was applied to the training dataset to ensure that all three subsets are represented more uniformly during training, regardless of their original sample counts. This helps prevent the model from being biased toward overrepresented cases and promotes more balanced learning across different flow scenarios.  

The U-Net consists of an encoder–decoder structure with skip connections to preserve spatial resolution. Each encoder block is composed of two successive convolutional layers with kernel size 3, followed by batch normalization and ReLU activation. Downsampling is performed using max-pooling layers, progressively reducing spatial resolution while increasing feature dimensionality (32 → 64 → 128 → 256 channels). At the bottleneck, the most compressed representation of the input is learned, capturing global flow structures such as wake dynamics and large-scale velocity patterns. The decoder mirrors the encoder structure using transposed convolutions for upsampling. At each resolution level, feature maps from the encoder are concatenated with the corresponding decoder features via skip connections. This allows the network to recover fine-scale spatial details such as boundary layers and sharp gradients. The final output layer is a 1×1 convolution that maps the features to the predicted velocity components.

The model is trained using a normalized mean squared error (NMSE) loss, which normalizes by the true field magnitude so all cases contribute proportionally to the loss regardless of their absolute velocity scale. This gave better results compared to a mean squared error (MSE) loss, which prioritized the minimization of errors on high-velocity cases because those have large absolute values and dominate the MSE.

$$
\mathrm{NMSE} = \frac{\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2}{\sum_{i=1}^{n} Y_i^2}
\qquad
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

The network is optimized using the Adam optimizer with a learning rate of $\text{lr} = 5 \times 10^{-4}$. Training is performed for up to 300 epochs, with early stopping based on validation loss to prevent overfitting. A patience parameter of 80 epochs is used, meaning training is terminated if no improvement in validation performance is observed within this window. At each epoch, the model is evaluated on both training and validation datasets. The model state corresponding to the lowest validation loss is saved, along with normalization statistics used for inputs and outputs to ensure consistent preprocessing during inference. The training process is performed on GPU when available. The dataset is loaded using mini-batch training with a batch of 32, and performance is monitored through epoch-wise tracking of training and validation losses. The final model is selected based on the best validation performance.

## Results
As stated previously, two separate models were developed for different prediction purposes. The first model is trained to predict the immediate next time step, while the second model is designed to predict the flow field five time steps ahead. Focusing on the one-step-ahead model, the training and validation loss curves exhibit a consistently decreasing trend, indicating that the network learns effectively throughout the training process. Furthermore, the close agreement between the training and validation losses suggests that the model does not overfit the training data and is able to generalize well, rather than simply memorizing the training samples. 

<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/397104f4-6571-4974-95a5-d31253e8ad8f" />

In the table below the mean and max relative errors across all testing samples are depicted, showing excellent generalization ability of the model to unseen data.

| Single-Timestep Prediction | U | V 
| :---: | :---: | :---: |
| **Mean Relative Error (%)** | 0.34 | 0.38 
| **Max Relative Error (%)** | 1.14 | 1.05 

To contextualize the model's performance, predictions are compared against an identity baseline, which assumes no change between consecutive timesteps (i.e., u(t+1) = u(t)). This serves as 
a lower bound for meaningful prediction — any useful model should outperform it. The identity baseline error reflects how much the flow actually changes per timestep. For steady flows (Re < 47), consecutive frames are nearly identical, making the identity an almost perfect predictor. For flows in the active vortex shedding regime (Re > 100), the flow evolves significantly between timesteps, and the identity baseline becomes less accurate. The table below compares the mean relative error of the model against the identity baseline across different Reynolds number ranges:

| Reynolds Range | Model Error (%) | Identity Error (%) | Improvement (%)
| :---: | :---: | :---: | :---:|
| 0-50 | 0.11 | 0.0 | -0.11 
| 50-100 | 0.12 | 0.0 | -0.12
| 100-150 | 0.27 | 0.23 | -0.04
| 150-200 | - | - | - 
| 200-250 | 0.22 | 0.47 | +0.25
| 250-300 | 0.22 | 0.46 | +0.25 
| 300-350 | 0.36 | 0.67 | +0.31
| 350-400 | 0.45 | 0.65 | +0.20
| 400 | 0.57 | 0.63 | +0.06

The model outperforms the identity baseline in the active vortex shedding regime (Re > 200), achieving up to 0.31% improvement in mean relative error. For low Reynolds number flows (Re < 150) where the flow is steady or transitional, the identity baseline remains more accurate — the model introduces small unnecessary perturbations where no meaningful dynamics exist. This suggests the model is specifically effective for capturing periodic vortex dynamics in the moderate-to-high Reynolds number regime. The plots below compare the true and predicted velocity fields for both the x- and y-components, along with the corresponding absolute error, for three representative test samples. The model accurately captures key flow features including the boundary layer around the cylinder, the recirculation wake, and the Kármán vortex street at higher Reynolds numbers. As expected, the largest errors are concentrated in the near-cylinder region and the wake, where velocity gradients are steepest and flow dynamics are most complex. Nevertheless, even in these challenging regions the absolute error remains low relative to the local velocity magnitude.

<img width="1836" height="1030" alt="image" src="https://github.com/user-attachments/assets/388d90e0-3fc2-449f-9a90-1d8bde6c97df" />
<img width="1846" height="1017" alt="image" src="https://github.com/user-attachments/assets/eacd653f-4367-4f26-91d3-b2c342fdb26f" />
<img width="1820" height="1016" alt="image" src="https://github.com/user-attachments/assets/2520e194-7060-4604-97df-eb6486687392" />

Building on the one-step-ahead model, a second autoregressive model was developed to predict the flow field five simulation timesteps ahead. As shown by the identity baseline analysis, consecutive frames differ only marginally - the flow changes by less than 0.7% per timestep even in the active vortex shedding regime. Predicting five steps ahead increases the temporal gap between input and target, resulting in larger flow field differences and a more challenging prediction task. This is particularly relevant for unsteady flows at higher Reynolds numbers, where the Kármán vortex street introduces significant frame-to-frame variation over longer time horizons. The five-step model therefore provides a more meaningful test of the network's ability to capture genuine flow dynamics, rather than simply predicting small incremental changes.

Once again, the training and validation loss curves confirm that the model learns effectively, with both losses decreasing consistently throughout training and insignificant overfitting. Losses are a bit higher than before, as expected.

<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/8a73a77a-d6c0-4834-beb6-acc6a596dbf0" />

In the table below the mean and max relative errors across all testing samples are depicted, showing once again excellent generalization ability of the model to unseen data. Error is higher than the single-timestep case, as expected.

| Single-Timestep Prediction | U | V 
| :---: | :---: | :---: |
| **Mean Relative Error (%)** | 0.74 | 0.78 
| **Max Relative Error (%)** | 2.69 | 2.56 

The five-step-ahead model achieves a mean relative error of 0.74% for both velocity components, with a maximum relative error of 2.69% and 2.56% for the x- and y-components respectively. As expected, these values are higher than the one-step-ahead model (0.37% mean relative error), reflecting the increased difficulty of predicting further into the future. The identity baseline comparison reveals a significantly clearer improvement over the baseline compared to the one-step model. In the active vortex shedding regime (Re > 200), the model reduces the error by up to 2.54% relative to the identity baseline, which itself has errors of 2.3-3.3% over five timesteps. This represents a 3-4x improvement over the identity prediction, confirming that the network genuinely learns the flow dynamics rather than simply persisting the current state. For low Reynolds number flows (Re < 100) the model again performs worse than the identity baseline, consistent with the one-step model findings - steady flows require no prediction and the network introduces unnecessary perturbations. 

| Reynolds Range | Model Error (%) | Identity Error (%) | Improvement (%)
| :---: | :---: | :---: | :---: |
| 0-50 | 0.19 | 0.0 | -0.19 
| 50-100 | 0.18 | 0.001 | -0.18
| 100-150 | 0.64 | 1.15 | +0.51
| 150-200 | - | - | - 
| 200-250 | 0.48 | 2.34 | +1.86
| 250-300 | 0.38 | 2.29 | +1.90 
| 300-350 | 0.78 | 3.33 | +2.54
| 350-400 | 0.93 | 3.17 | +2.24
| 400 | 1.30 | 3.09 | +1.79

Three representative test samples are again evaluated against the ground truth and visualized below. Despite the increased prediction horizon of five timesteps, the model continues to capture the key flow features accurately, including the cylinder boundary layer, the recirculation wake, and the Kármán vortex street. As expected, the absolute error is slightly higher than the one-step-ahead model, particularly in the near-cylinder region and the wake where the flow evolves most rapidly over the five-timestep interval. Nevertheless, the overall flow structure is well reproduced across all tested cases.

<img width="1849" height="1014" alt="image" src="https://github.com/user-attachments/assets/a9f0eef4-383c-4b21-96ca-d596ea219b3d" />
<img width="1844" height="1008" alt="image" src="https://github.com/user-attachments/assets/8be6d2bc-728e-445c-aff6-73398c967838" />
<img width="1840" height="1018" alt="image" src="https://github.com/user-attachments/assets/fc667bd5-d1d6-442f-a8b6-28677ce03d5c" />

## Conclusion & Future Work
This study investigated the use of an autoregressive U-Net surrogate model for predicting the temporal evolution of flow around a circular cylinder across varying boundary conditions, geometric configurations, and fluid properties. Two models were developed and evaluated: a one-step-ahead model predicting the immediate next simulation frame, and a five-step-ahead model predicting the flow field five timesteps into the future. Both models demonstrate effective learning of the underlying flow dynamics, with training and validation losses converging consistently and negligible overfitting observed in both cases. The one-step-ahead model achieves a mean relative error of 0.37% for both velocity components, while the five-step model achieves 0.74% - a modest increase reflecting the inherently more challenging prediction horizon. Comparison against an identity baseline confirms that both models genuinely capture flow dynamics rather than simply persisting the current state, with improvements of up to 0.31% and 2.54% respectively in the active vortex shedding regime (Re > 200). For low Reynolds number flows (Re < 100) where the flow is steady, the identity baseline remains more accurate - an expected result given the absence of meaningful temporal dynamics to learn. The results are consistent with findings reported in CFDBench (Luo et al., 2023), which identified cylinder flow as one of the most challenging problems for autoregressive neural operators due to the complexity of periodic vortex shedding dynamics.

Regarding relevant future work, several directions could extend and improve upon the present work:

* Extended input representation. The current model uses Reynolds number and a binary cylinder mask as input channels. Following the CFDBench methodology more closely, individual physical parameters (inlet velocity, density, viscosity, radius) could be used as separate input channels, potentially improving generalization across the full parameter space.
* Larger and more diverse datasets. The current study used a filtered subset of CFDBench cases (Re ≤ 400, density ≤ 20). Training on the full parameter range, including high Reynolds number turbulent cases, would test the model's generalization to more complex flow regimes.
* Rollout evaluation with alternative architectures. While the U-Net demonstrated strong single-step prediction accuracy, CFDBench (Luo et al., 2023) reports significant error accumulation for U-Net during multi-step autoregressive rollout - where the model's own predictions are fed back as inputs over many steps. Alternative architectures such as the Fourier Neural Operator (FNO) or recurrent networks (e.g. ConvLSTM) may be better suited for stable long-horizon rollout, as they have shown more favorable error accumulation properties in the benchmark.
* The purely data-driven U-Net surrogate has no explicit knowledge of the governing Navier-Stokes equations - it learns flow dynamics entirely from data. A promising direction is to combine the U-Net architecture with physics-informed constraints, embedding the residuals of both the continuity and momentum equations directly into the loss function. The continuity equation enforces mass conservation (∇·u = 0), while the momentum equations ensure physically consistent force balances at each predicted timestep. Such a hybrid CNN-PINN approach would constrain the network to produce physically consistent predictions, potentially improving accuracy in data-sparse regimes and reducing error accumulation during autoregressive rollout. One practical challenge is that enforcing the momentum equations requires knowledge of the pressure field, which is not currently predicted by the model - this could be addressed by extending the network output to include pressure.

## References
Luo, Y., Chen, Y., & Zhang, Z. (2023). CFDBench: A large-scale benchmark for machine learning methods in fluid dynamics. (https://arxiv.org/abs/2310.05963). 

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Proceedings of the 18th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015). [https://arxiv.org/abs/1505.04597.](https://arxiv.org/abs/1505.04597v1).  

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. Proceedings of the 32nd International Conference on Machine Learning (ICML 2015). https://arxiv.org/abs/1502.03167.  

Shibuya, E., & Hotta, K. (2021). Cell image segmentation by using feedback and convolutional LSTM. The Visual Computer, 38(12). https://doi.org/10.1007/s00371-021-02221-3.  

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1412.6980.  

</div>
