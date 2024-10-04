# Inferring Transmission Rate Dynamics through Model Learning and Data Assimilation
This repository contains the code used to generate the results presented in [1]. The proposed architecture is inspired by a scientific machine learning method (LDNets) designed to uncover low-dimensional intrinsic dynamics in systems displaying spatio-temporal behavior in response to external stimuli. We only consider time dependence, since we are working with standard homogeneous compartmental models which are commonly used for making reliable forecasts in epidemic problems. Additionally, we incorporate Data Assimilation techniques to identify latent parameters on which the dynamics of the parameter to discover depends. Our goal is to learn the unknown dependencies governing the dynamics of transmission mechanisms during epidemic events. Using a compartmental model, we propose an innovative approach to extrapolate beyond the observed scenario, leveraging deterministic differential models.

## Requirements

To run these codes, you need Python (version 3.9) with the modules listed in the file `requirements.txt`. If you are using `pip`, you can install them by running:
```bash
pip install -r requirements.txt
```

## References

[1] Paper to submit.
