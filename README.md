# Master's Thesis on Bayesian Deep Learning - 2024/2025

This repository contains a work-in-progress implementation of my Master's thesis project, which will investigate how Bayesian last-layer neural network architectures can be leveraged to improve existing methods for explaining uncertainty. The primary objective will be to adapt an influential method of explaining uncertainty, CLUE (Counterfactual Latent Uncertainty Explanations)[^1], such that it exploits the architecture of Bayesian last-layer neural networks. The proposed method will align uncertainty explanations with the internal representations of the predictive model while reducing computational overhead.

I am pursuing this project in fulfilment of my Master's degree in Computer Engineering at Trinity College Dublin, from which I will graduate in Autumn 2025. This project will be completed by April 2025, at which point I will publish my thesis write-up here.

Currently, this README contains a short description of my project idea/proposal. I will update it with implementation details as I progress.

## Background and Motivation

### Bayesian Last-Layers
Approximating Bayesian inference in neural networks is computationally challenging due to the intractability of the posterior distribution and the high dimensionality of parameter spaces, making scalable and efficient methods essential for the wider adoption of Bayesian neural networks. 

Bayesian last-layer architectures represent a promising avenue for balancing computational efficiency and accuracy in uncertainty estimation. These architectures separate representation learning from uncertainty estimation by using deterministic layers to extract features and reserving Bayesian inference for the final layer. This hybrid approach significantly reduces the computational overhead associated with full-network Bayesian inference while preserving high-quality uncertainty estimates.

The paper “On Last-Layer Algorithms for Classification: Decoupling Representation from Uncertainty Estimation”[^2] demonstrated that Bayesian last-layer architectures achieve comparable performance to fully Bayesian networks tasks which involve quantifying uncertainty, such as out-of-distribution detection and selective classification. These findings suggest that uncertainty quantification primarily depends on the final layer of the network, making it unnecessary to apply Bayesian inference across all layers. Since the publication of this paper, there have been several more approaches proposed for this sort of partially Bayesian inference, such as a last-layer variant of the Laplace approximation[^3] and "Variational Bayesian last-layers"[^4].

### Explanations for Uncertainty
Explaining uncertainty in Bayesian neural networks is essential for making machine learning models more interpretable and trustworthy, especially in safety-critical applications like healthcare and autonomous systems. While uncertainty estimates themselves provide insights into a model’s confidence, methods like CLUE go further by seeking to identify the _sources_ of uncertainty.

CLUE works by generating counterfactual examples—alternative data points that are similar to the original input but lead to significantly more confident predictions. The difference between the original input and its counterfactual provides an interpretation of the source of uncertainty in the model's prediction. CLUE achieves this by leveraging a deep generative model (DGM), such as a Variational Autoencoder (VAE), trained on the same dataset as the Bayesian classifier. The method optimises in the VAE's latent space, ensuring that the generated counterfactuals remain realistic and within the data distribution.

Several approaches have been proposed since this paper was published, some of which build on CLUE while others take an alternative approach focused on feature attribution. For example, $\Delta$-CLUE[^5] generates diverse sets of counterfactual explanations, while a method based on path integrals[^6] attributes uncertainty to individual input features by analysing changes along the CLUE optimisation path. Feature attribution techniques, such as UA-Backprop[^7] and adaptations of existing XAI methods like LIME and LRP[^8], directly assign uncertainty contributions to specific input features without requiring an auxiliary generative model. These approaches are more lightweight than the CLUE-like methods, but their local explanations can have limited expressiveness and lack the same insights into how to _reduce_ uncertainty.

## Proposed Approach: Leveraging Bayesian Last-Layers for CLUE
The primary objective of this project is to adapt the CLUE method such that it exploits the architecture of Bayesian last-layer neural networks. More specifically, we aim to eliminate the reliance on the latent space of an external DGM by directly utilising the classifier's internal latent space for counterfactual generation. 

### How It Works
<div align="center">
  <img src="https://github.com/user-attachments/assets/4c894c0a-7e39-4c8f-92c5-e8b891f5622d" alt="61bdffca-432b-4d96-a558-c07c14099ed3 sketchpad" width="700" style="border:5px solid #000; border-radius:50px;">
</div>

1. **Classifier Training**:
   - Train a last-layer Bayesian neural network where the deterministic layers act as a feature extractor, mapping inputs \( x \) to an intermediate latent representation \( z \), and the Bayesian last layer provides uncertainty estimation.

2. **Latent Space Optimization**:
   - When encountering an uncertain prediction, the intermediate latent representation \( z \) is optimized to reduce uncertainty in the Bayesian last layer by:
     - Minimizing uncertainty in the classifier’s prediction.
     - Maintaining proximity to the original latent representation to preserve the input’s integrity.
   - By optimizing directly in the classifier’s latent space, the counterfactual search aligns closely with the model’s 'understanding' of the data.

3. **Final Counterfactual Generation**:
   - After latent space optimization, the final latent point is decoded using a DGM trained to reconstruct the training data from the features learned by the Bayesian neural network.
  
#### **Key Advantages**
This method focuses the optimization process on the classifier’s _internal_ latent space:
- **Efficiency**: Computation costs are reduced: instead of requiring $n$ generations and $n$ predictions
for $n$ steps of CLUE’s optimisation, only $n$ last-layer predictions are needed, followed by a single generation for the final counterfactual.
- **Alignment**: By using the classification latent space, the optimisation process in CLUE can focus on the classifier’s interpretation of the data, aligning the counterfactuals more closely with the model’s decision-making process.
- **Class-consistency**: As discussed in the following section, the latent space of the classifier is expected to be sparser than that of a deep generative model (DGM), with distinct separation between classes. We anticipate that this structure will result in counterfactuals that remain within the class of the original input. This behavior contrasts with the original implementation of CLUE, where crossing class boundaries was possible and could not be reliably avoided. We argue that counterfactuals which cross class boundaries provide an inferior explanation of uncertainty, as they conflate the explanation of uncertainty with an explanation of the classifier's underlying decision boundaries.

### Understanding the new latent space

<div align="center">
  <img width="500" alt="Screenshot 2025-01-15 at 18 13 37" src="https://github.com/user-attachments/assets/0ddd3dff-fdb4-4be7-9c74-3d631ea037e6" />
</div>

Although generating an input for each step of CLUE optimisation is no longer needed, it will still be necessary to generate an input for the final latent point. The paper ”Classify and generate: Using classification latent space representations for image generations”[^9] provides helpful guidance in this regard, as it describes how to use features extracted by a classification model for the downstream task of generation. As can be seen in the diagram above (taken from this paper), a classifier's latent space is optimised for separating classes and is typically sparser, with less emphasis on maintaining a continuous or well-structured manifold. 

The paper proposes techniques for remaining on the data manifold when generating data points which may prove useful for keeping our counterfactuals in-distribution. The hope is that their techniques won’t be necessary; if uncertainty is higher in sparser areas of the latent space, then the CLUE algorithm will naturally ’find’ its way back to the data manifold.

## **Secondary Task: Exploring Adjacent Methods**

While the proposal described above is likely to provide some interesting results, it is anticipated that there will be some limitations relating to reconstruction accuracy from the classification latent space for some datasets. This is why, subject to progress on the primary objective, this dissertation also aims to explore how insights from adjacent methods can be applied in this context:

- **Path Integrals [^6]**: By analyzing the changes in input dimensions during CLUE optimization, this approach could help mitigate challenges like poor reconstruction accuracy.
- **UA-Backprop [^7]**: The uncertainty gradients backpropagated in this method are not decomposable into epistemic and aleatoric uncertainties, as the set of gradients obtained for each sampling of the weights is averaged post-hoc. This may change for last-layer models, where most of the layers are fixed.


# **Conclusion**

In conclusion, this project will investigate how Bayesian last-layer neural network architectures can be leveraged to improve existing methods for explaining uncertainty. By adapting CLUE to last-layer Bayesian architectures, we aim to improve computational efficiency, align counterfactuals with the classifier’s internal representations, and generate counterfactuals which are consistent with the class of the original input. The integration of adjacent methods, such as path integrals and UA-Backprop, may enhance the robustness of the proposed method.


[^1]: Antor'an, Javier et al. “Getting a CLUE: A Method for Explaining Uncertainty Estimates.” ArXiv abs/2006.06848 (2020): n. pag.
[^2]: Brosse, Nicolas et al. “On Last-Layer Algorithms for Classification: Decoupling Representation from Uncertainty Estimation.” ArXiv abs/2001.08049 (2020): n. pag.
[^3]: Daxberger, Erik A. et al. “Laplace Redux - Effortless Bayesian Deep Learning.” Neural Information Processing Systems (2021).
[^4]: Harrison, James et al. “Variational Bayesian Last Layers.” ArXiv abs/2404.11599 (2024): n. pag.
[^5]: Ley, D. et al. “Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates.” ArXiv abs/2112.02646 (2021): n. pag.
[^6]: Perez, Iker et al. “Attribution of predictive uncertainties in classification models.” Conference on Uncertainty in Artificial Intelligence (2021).
[^7]: Wang, Hanjing et al. “Gradient-based Uncertainty Attribution for Explainable Bayesian Deep Learning.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023): 12044-12053.
[^8]: Brown, Katherine E. and Douglas A. Talbert. “Using Explainable AI to Measure Feature Contribution to Uncertainty.” The International FLAIRS Conference Proceedings (2022): n. pag.
[^9] Gopalakrishnan, Saisubramaniam et al. “Classify and generate: Using classification latent space representations for image generations.” Neurocomputing 471 (2020): 296-334.

