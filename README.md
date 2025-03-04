# Counterfactual Explanations from Internal Representations

This year I'm working on a research project in the area of Bayesian Deep Learning as part of my Master's degree in Computer Engineering. It's still a work-in-progress, but I've included a short summary of where the project is currently at below. The scope of the project may change over time, so I'll try to update this often. I'll certainly make the dissertation avilable once I complete it in late April.


## Counterfactual Explanations for Black-box Models
The field of Explainable AI (XAI) is vast and contains many sub-fields and approaches to explaining the predictions of machine learning models. Within the sub-field of explanations for black-box models (e.g. neural networks), one popular approach is to use *counterfactual explanations*. If we take as an example a neural network that decides whether you're eligible for a loan, and the network predicts that you are not eligible, a counterfactual explanation would give you a minimal change to the data (e.g. increase your income by €10,000) which would result in the network instead predicting that you *are* eligible.

Counterfactual explanations are very actionable and interpretable, but come with some difficulties when compared with other approaches like feature attributions (e.g. which one of my features were most important in denying me a loan, my income, age, or credit score?). The central challenge is staying *on the data manifold* when generating counterfactuals. This refers to the fact that the perturbations made to the data should produce counterfactuals that are still valid data points (e.g. don't tell me to increase my income by €10,000,000).

In practice, most counterfactual explanation methods leverage a deep generative model to find counterfactuals. This generative model is trained on the same data as the classification model we want to explain. To produce a counterfactual, we encode the original data point into the latent space of the generative model, and then take minimal steps within the latent space to optimise the objective. In our previous example, the objective would be to change the predicted class from not eligible to eligible. There are two main problems with this approach:

1. For $n$ optimisation steps to find a counterfactual, we need to make $n$ forward passes through the generative model. This can be quite slow for complex datasets and models.
2. The latent space of the generative model is not aligned with the learned representations of the classification model. This means that the counterfactuals generated may be adversarial examples (i.e. highly irregular outliers to the underlying decision-making process of the classifier) and thus don't truly explain the local decision boundary.
3. The quality of our explanation process for the classification model is dependent on the quality of the generative model. For complex problems where generating high-quality samples is difficult, this is a challenge.

In a way, this approach of using a generative model to produce counterfactuals is overkill. We don't truly need the capacity to sample new data points from the input distribution, we simply need to find meaningful changes to existing data. In this sense, we are only using the generative model to approximate the data manifold. However, as we'll see later, there are other proxies for the data manifold that we can leverage to this end.

## Bayesian Neural Networks
Bayesian Neural Networks (BNNs) are a special type of neural network that is very good at providing uncertainty estimates for their predictions. Where traditional neural networks are usually overconfident about incorrect predictions, BNNs are much more accurate in estimating their uncertainty. This sort of behaviour is essential in situations where incorrect predictions can have dire consequences, such as in healthcare or autonomous driving. 

BNNs achieve this by learning a distribution over the weights of the network, rather than a single point estimate. This allows them to provide a distribution over their predictions, which can be used to estimate the epistemic uncertainty (uncertainty in the model's ability to generalise) and aleatoric uncertainty (inherent and irreducible uncertainty in the data) of their predictions.

BNNs have been shown to have statistical guarantees for robustness to adversarial attacks, where input data is altered in such a way that is undetectable by humans, but causes the neural network to change it's prediction. Additionally, they are excellent at identifying out of distribution data i.e. they can identify outliers in the data at inference time that are not representative of the training data. One of the goals of this dissertation is show how this means that Bayesian Neural Networks construct an implicit approximation to the data manifold.

### Bayesian Last-Layer Neural Networks
While Bayesian neural networks have many advantages over traditional neural networks, the approximate Bayesian inference required to learn the weight distributions is computationally expensive. One proposed avenue for reducing this computational burden is to focus all of the uncertainty estimation in the final layer of the network. That is, we train a regular neural network, and then "chop off" the final layer and train a single Bayesian layer on the penultimate layer's representations. 

It has been shown that these sorts of hybrid models can achieve almost identical performance to fully Bayesian equivalents on uncertainty-related metrics such as out-of-distribution detection and calibration.

## Bringing It All Together
Recent work has demonstrated a method of generating realistic counterfactuals without the need for a generative model[^2]. By using deep ensembles (approximately Bayesian) to estimate epistemic and aleatoric uncertainties, they perform a search process *directly in the input space* to find counterfactuals. By guiding the input space perturbations with the uncertainty estimates, they are able to generate remarkable realistic and in distribution counterfactuals. 

This being said, the counterfactuals are still of lower quality to those generated using a generative model. While the uncertainty estimates do an impressive job of keeping the search process close the data manifold, the unconstrained and high-dimensional input space is still a challenge. 

### This Dissertation's Proposal
This brings us to the central proposal of this dissertation. What if, rather than performing an uncertainty-guided search process in the input space, we could perform a similar search process in the internal latent space of the classification model? This lower-dimensional space is much more constrained, is aligned with how the model actually makes decisions, and directions in this space will correspond to meaningful changes to the input data. However, if we explore the internal representations of a fully Bayesian model (or a deep ensemble like the one used in the previous work), we will lose the necessary uncertainty estimation provided by the earlier layers in the model.

That is, unless we use a *Bayesian last layer*. In this case, we can simply perturb the representation that lies at the penultimate layer of the model, and observe the effects on the uncertainty-aware prediction of the final layer. This way, we can explore the latent space of the model, while staying close to the data manifold.

At first, the goal of our counterfactual explanations will be to explain the *uncertainty* of the model's predictions, rather than changing the prediction to something else. In our loan example, this could be "Improve your credit score by 10 points to increase our confidence in granting a loan from 55% to 90%". This is an approach taken by CLUE (Counterfactual Latent Uncertainty Explanations)[^1] which use a deep generative model, as described in the first section, to find minimal changes to an input which *reduce* the model's uncertainty.

The final issue to address is how we can make these latent space changes *interpretable*. In practice, this is a question of how we project these changes back onto the input space so users can understand the changes in the input data necessary to reduce the model's uncertainty. We set out two primary methods for doing this:

**1. Decoder**: The first and most straightforward approach is to train a decoder to reconstruct inputs from the latent space of the classifier. Granted, this essentially reintroduces the need for a generative model, but we will have removed the generative model's role in the counterfactual search process. Also, in this case, it is more of an autoencoder than a variational autoencoder, and thus is optimised for reconstructing the training data and not for sampling new data from the training distribution. 
- Initial experiments show that using this setup with a traditional neural network (i.e. guiding the search process with the deterministic entropy of the final layer) finds adversarial latent points $z$ which results in 0 entropy in the final layer, but decode to form *the exact same input as the original data*. So, if the decoded counterfactual is passed through the model again, the uncertainty will be unchanged.
- If further experiments with a Bayesian last layer instead result in *actual changes* to the input data which reduce uncertainty when passed through the model again, we will have some evidence that the Bayesian last layer is able to guide the search process to find latent points $z$ which correspond to real input data points.

**2. Path Integrals**: Next, I plan on developing an approach based on path integrals, inspired by the integrated gradients (INTGRAD) method. The original version of this XAI method involves selecting a "null" reference input (e.g. a black image) and gradually transforming this into the target input. The gradients of the prediction with respect to the input features at each step are then integrated to produce a saliency map, showing how each feature contributed to the final prediction.
- This general framework has since been applied to the task of explaining uncertainty[^3]. In this paper, the authors instead leverage a generative model (like in CLUE)[^1] to find a counterfactual, but then use the path integral between the original and counterfactual latent points to produce a saliency map.
- Applied to our case, we would instead use the path integral between the original and counterfactual latent points at the penultimate layer of the model to project the changes back onto the input space. The drawback here is that we are now provide *feature attributions* rather than counterfactual explanations. 

Early experiments of the first method have indeed shown that Bayesian last layers are able to guide the search process in such a way that results in latent points $z$ which decode to form input data points which meaningfully differ from the original input. When the decoded input is passed through the entire model again, the uncertainty is significantly reduced. I'm currently working on some larger scale experiments to observe the extent of this behaviour, but these initial signs are encouraging. 

The central goal of the project is to fully implement and test both of the above methods, using their performance to analyse how Bayesian last layers implicitly learn a representation of the data manifold within the penultimate layer representations of the model. If these methods are successful, I have also proposed a third method which allows the path integrals approach to be used to actually generate counterfactuals. This method is described in the following appendix, should it be of interest.


### Appendix: Generating Counterfactuals with Path Integrals
Let’s reconsider the paper where counterfactual explainations are generated without a generative model by guiding an input space search with uncertainty estimates. The authors use a simple implementation of a saliency map (i.e. the gradient of the prediction with respect to the input features) to iteratively update the values of inputs dimensions, while using the epistemic and aleatoric uncertainties to keep them on manifold. 

However, methods based on integrated gradients claim to produce better saliency maps and, in the previous section, we have proposed a method of integrating gradients along an optimisation path in the classification latent space to produce even better saliency maps. What if we used *these* saliency maps for the input dimension updates? Since the inputs will need several updates to move towards a counterfactual, this would involve a sort of meta-optimisation approach with an inner and an outer loop:

1. **Inner Loop: Latent Space Optimization** 
    1. Start with the original input  $x_0$ , which maps to latent representation  $z_0$
    2. Perform iterative **latent space optimization** to find a low-entropy latent point  $z^*$. 
    3. Track the **gradient of latent updates** with respect to the input space: $\frac{\partial z}{\partial x}$
    4. Store the **full trajectory of updates**  $(z_0 \to z_1 \to \dots \to z^*)$.
2. **Integrate Latent Step Gradients Back to Input Space**
    1. Instead of using the final optimized  $z^*$ , **integrate the step-wise gradients back to the input space**, similar to **Integrated Gradients (INTGRAD)**: $\Delta x = \sum_{t=0}^{T} (z_{t} - z_{t-1}) \cdot \frac{\partial z}{\partial x} \Big|_{z_t}$
    2. This forms a **saliency map** indicating how each input feature contributed to the uncertainty reduction.
3. **Outer Loop: Input Space Update**
    1. Update  $x$  using this **feature attribution map**: $x{\prime} = x + \alpha \cdot \Delta x$
    2. Here, $\alpha$ is a step size controlling how much we modify the input.
    3. This ensures that input-space changes are informed by **what actually reduced uncertainty in latent space**.
4. **Iterate Until Convergence**
    1. Recalculate the latent representation for the updated input $x{\prime}$, then repeat the process.
    2. Stop when:
        1. The classifier’s predictive entropy is sufficiently low.
        2. The change in $x$ stabilizes.

It's unlikely that it will be possible to implement this method within the scope of this dissertation, but I plan on having a go at it once the project is complete.


[^1]: Antorán, Javier et al. “Getting a CLUE: A Method for Explaining Uncertainty Estimates.” ArXiv abs/2006.06848 (2021): n. pag.
[^2]: Schut, Lisa et al. “Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties.” ArXiv abs/2103.08951 (2021): n. pag.
[^3]: Perez, Iker et al. “Attribution of Predictive Uncertainties in Classification Models.” ArXiv abs/2107.08756 (2022): n. pag.

















