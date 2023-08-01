# Learning Probability Density Functions from Samples

Small experiment to see if it's possible to learn the PDF of an arbitrary distribution given some samples from it.

Below: a SIREN network learns the continuous PDF of two gaussians.

- **Input:** A discrete PDF generated from the samples.

- **Output:** A SIREN network that parameterizes the continuous PDF. To demonstrate this, I sampled the discrete PDF domain at 10x resolution, but there is no limit to how finely you can sample probabilities from the SIREN.

![data sample](readme_assets/data.png)

![discrete pdf generated from samples](readme_assets/training_density_map.png)

![learned continuous pdf using siren](readme_assets/predicted_density_map.png)
