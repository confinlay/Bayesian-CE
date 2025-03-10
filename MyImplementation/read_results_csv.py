import pandas as pd
import numpy as np
import os

def summarize_model_results(csv_path, model_name):
    """
    Reads a CSV file with the following tab‐separated format:
    
      image_index  latent_distance  original_entropy_latent  counterfactual_entropy_latent  latent_entropy_reduction  original_entropy_recon  counterfactual_entropy_recon  recon_entropy_reduction  original_log_likelihood  reconstruction_log_likelihood  counterfactual_log_likelihood  log_likelihood_difference  ... 
      0   4.776182651519780   0.5305212736129760   0.004766764119267460  0.5257545113563540   0.7066278457641600   0.49189358949661300  0.2147342562675480   -67.4476318359375   -73.54015350341800   -73.57111358642580  6.123481750488280  ...
      1   6.220684051513670   1.2779650688171400   0.005583701189607380  1.2723814249038700   0.9731091260910030   0.48012062907218900  0.4929884970188140   -88.86124420166020   -94.84294128417970   -92.85356140136720  3.992317199707030  ...
      ...
      
    It then computes the following aggregate metrics over the file:
    
      - avg_recon_entropy_reduction: mean of the "recon_entropy_reduction" column.
      - Average NLL difference (reconstructions): mean of "log_likelihood_difference".
      - Original latent entropy: mean of "original_entropy_latent".
      - Reconstruction entropy: mean of "original_entropy_recon".
      - Counterfactual Entropy: mean of "counterfactual_entropy_recon".
      - Class Agreement (original vs. Counterfactual): fraction of rows where 
        "original_class_latent" equals "counterfactual_class_latent".
      - Original log likelihood: mean of "original_log_likelihood".
      - Original reconstruction log likelihood: mean of "reconstruction_log_likelihood".
      - Counterfactual reconstruction log likelihood: mean of "counterfactual_log_likelihood".
    
    The model name is set by the user since the entire file represents one model.
    
    Parameters:
      csv_path (str): Path to the CSV file.
      model_name (str): Name of the model for labeling the summary.
      
    Returns:
      pd.DataFrame: A one-row DataFrame with the summary statistics.
    """
    # Read the CSV file (assuming tab-separated)
    df = pd.read_csv(csv_path, sep=",")

    print("Available columns:", df.columns.tolist())

    
    summary = {
        "Model": model_name,
        "avg_recon_entropy_reduction": df["recon_entropy_reduction"].mean(),
        "Average NLL difference (reconstructions)": df["log_likelihood_difference"].mean(),
        "Original latent entropy": df["original_entropy_latent"].mean(),
        "Reconstruction entropy": df["original_entropy_recon"].mean(),
        "Counterfactual Entropy": df["counterfactual_entropy_recon"].mean(),
        "Class Agreement (original vs. Counterfactual)": (df["original_class_latent"] == df["counterfactual_class_recon"]).mean(),
        "Original log likelihood": df["original_log_likelihood"].mean(),
        "Original reconstruction log likelihood": df["reconstruction_log_likelihood"].mean(),
        "Counterfactual reconstruction log likelihood": df["counterfactual_log_likelihood"].mean()
    }
    
    # Create a one-row DataFrame from the summary dictionary
    summary_df = pd.DataFrame([summary])
    
    # Save the summary to a CSV file in the same directory as the input file
    import os
    output_dir = os.path.dirname(csv_path)
    output_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_summary.csv")
    summary_df.to_csv(output_filename, index=False)
    
    return summary_df



# Example usage:
if __name__ == '__main__':
    print(os.getcwd())
    os.chdir("/Users/conor/Documents/College terms/College/Thesis/Thesis_Code_Minimised/MyImplementation")
    model = "joint_model_256_same_pics_2025-03-08_19-47-56"
    csv_path = f"model_saves/new_regene_models/CLUE_results/{model}/individual_results.csv"
    model_name = "Joint model, Bayesian, same pics"  # Set the model name as desired
    summary_table = summarize_model_results(csv_path, model_name)
    print(summary_table)
