# Master's Thesis on Bayesian Deep Learning - 2024/2025

This repository contains a work-in-progress implementation of my Master's thesis project, which will investigate how Bayesian last-layer neural network architectures can be leveraged to improve existing methods for explaining uncertainty. The primary objective will be to adapt an influential method of explaining uncertainty, CLUE (Counterfactual Latent Uncertainty Explanations)[^1], such that it exploits the architecture of Bayesian last-layer neural networks. The proposed method will align uncertainty explanations with the internal representations of the predictive model while reducing computational overhead.

I am pursuing this project in fulfilment of my Master's degree in Computer Engineering at Trinity College Dublin, from which I will graduate in Autumn 2025. This project will be completed by April 2025, at which point I will publish my thesis write-up here.

Currently this README just contains a short description of my project idea/proposal. I will update it with implementation details as I progress.

## Background and Motivation

### Bayesian Last-Layers
Approximating Bayesian inference in neural networks is computationally challenging due to the intractability of the posterior distribution and the high dimensionality of parameter spaces, making scalable and efficient methods essential for the wider adoption of Bayesian neural networks. 

Bayesian last-layer architectures represent a promising avenue for balancing computational efficiency and accuracy in uncertainty estimation. These architectures separate representation learning from uncertainty estimation by using deterministic layers to extract features and reserving Bayesian inference for the final layer. This hybrid approach significantly reduces the computational overhead associated with full-network Bayesian inference while preserving high-quality uncertainty estimates.

The paper “On Last-Layer Algorithms for Classification: Decoupling Representation from Uncertainty Estimation”[^2] demonstrated that Bayesian last-layer architectures achieve comparable performance to fully Bayesian networks tasks which involve quantifying uncertainty, such as out-of-distribution detection and selective classification. These findings suggest that uncertainty quantification primarily depends on the final layer of the network, making it unnecessary to apply Bayesian inference across all layers. Since the publication of this paper, there have been several more approaches proposed for this sort of partially Bayesian inference, such as a last-layer variant of the Laplace approximation[^3] and "Variational Bayesian last-layers'[^4].

### Explanations for Uncertainty
Explaining uncertainty in Bayesian neural networks is essential for making machine learning models more interpretable and trustworthy, especially in safety-critical applications like healthcare and autonomous systems. While uncertainty estimates themselves provide insights into a model’s confidence, methods like CLUE go further by seeking to identify the _sources_ of uncertainty.

CLUE works by generating counterfactual examples—alternative data points that are similar to the original input but lead to significantly more confident predictions. The difference between the original input and its counterfactual provides an interpretation of source of uncertainty in the model's prediction. CLUE achieves this by leveraging a deep generative model (DGM), such as a Variational Autoencoder (VAE), trained on the same dataset as the Bayesian classifier. The method optimizes in the VAE's latent space, ensuring that the generated counterfactuals remain realistic and within the data distribution.

There have been several approaches proposed since this paper was published, some of which build on CLUE while others take an alternative approach focused on feature attribution. For example, $\Delta$-CLUE[^5] generates diverse sets of counterfactual explanations, while a method based on path integrals[^6] attributes uncertainty to individual input features by analyzing changes along the CLUE optimisation path. Feature attribution techniques, such as UA-Backprop[^7] and adaptations of existing XAI methods like LIME and LRP[^8], directly assign uncertainty contributions to specific input features without requiring an auxiliary generative model. These approaches are more lightweight than the CLUE-like methods, but their local explanations can have limited expressiveness and lack the same insights into how to _reduce_ uncertainty.


[^1]: Antor'an, Javier et al. “Getting a CLUE: A Method for Explaining Uncertainty Estimates.” ArXiv abs/2006.06848 (2020): n. pag.
[^2]: Brosse, Nicolas et al. “On Last-Layer Algorithms for Classification: Decoupling Representation from Uncertainty Estimation.” ArXiv abs/2001.08049 (2020): n. pag.
[^3]: Daxberger, Erik A. et al. “Laplace Redux - Effortless Bayesian Deep Learning.” Neural Information Processing Systems (2021).
[^4]: Harrison, James et al. “Variational Bayesian Last Layers.” ArXiv abs/2404.11599 (2024): n. pag.
[^5]: Ley, D. et al. “Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates.” ArXiv abs/2112.02646 (2021): n. pag.
[^6]: Perez, Iker et al. “Attribution of predictive uncertainties in classification models.” Conference on Uncertainty in Artificial Intelligence (2021).
[^7]: Wang, Hanjing et al. “Gradient-based Uncertainty Attribution for Explainable Bayesian Deep Learning.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023): 12044-12053.
[^8]: Brown, Katherine E. and Douglas A. Talbert. “Using Explainable AI to Measure Feature Contribution to Uncertainty.” The International FLAIRS Conference Proceedings (2022): n. pag.
