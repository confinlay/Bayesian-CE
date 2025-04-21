# Uncertainty-Guided Counterfactual Explanations from Interal Representations

## Project Overview

This repository contains the implementation for my dissertation on Explainable AI and Counterfactual Explainability. It presents a novel approach to generating counterfactual explanations for deep neural networks using Bayesian uncertainty. Unlike traditional methods that operate in the input space or use auxiliary generative models, this approach leverages the discriminative latent space of the classifier itself.

## Core Concept

The key insight behind Bayesian-CE is that counterfactual explanations should be found within the model's own internal representations—specifically, the penultimate layer embeddings that the model uses to make its decisions. This approach aims to generate counterfactuals that are:

1. More semantically meaningful and relevant to the model's decisions
2. Better aligned with human notions of similarity
3. More reliable through principled uncertainty quantification

## Theoretical Background

Research suggests that the penultimate layer of a deep classifier contains rich, semantically meaningful representations organised around class prototypes. By perturbing these embeddings and using a Bayesian last layer to guide the optimisation, we can find counterfactuals that transform along decision-relevant features rather than adversarial directions.

## Methodology

Bayesian-CE performs gradient-based optimisation in the penultimate layer's latent space, guided by the Bayesian last layer's prediction confidence. The objective function balances:
- Maximising confidence in the target class
- Minimising distance from the original example in latent space

The optimised embedding is then decoded back to the input space to create a human-interpretable counterfactual.

## Architecture Components

The implementation consists of three core components:

1. **Classifier with Decodable Embeddings**: A jointly trained model that both classifies inputs and preserves information for reconstruction
2. **Bayesian Last Layer**: A variational Bayesian neural network layer that provides principled uncertainty quantification
3. **Counterfactual Optimiser**: A gradient-based optimiser that explores the latent space to find counterfactual explanations
