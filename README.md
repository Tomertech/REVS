# REVS: Unlearning Sensitive Information in LLMs

Welcome to the REVS (Rank Editing in the Vocabulary Space) repository. REVS introduces a novel technique aimed at unlearning sensitive information from Large Language Models (LLMs) while minimally degrading their overall utility. This method, detailed in our [REVS Paper](https://tomertech.github.io/REVS-Web/), is at the forefront of ensuring privacy and security in the deployment of LLMs. Currently, we support EleutherAI's GPT-J 6B and Llama 3 8B models.

Contributions, feedback, and discussions are highly encouraged. Should you face any challenges or wish to propose enhancements, please do not hesitate to open an issue.

![REVS Main Method](https://tomertech.github.io/REVS-Web/static/images/Main%20Method%20Plot%20Wide%20tinypng.png "REVS Main Method Overview")

*Editing one neuron with REVS:* (1) The neuron is projected from hidden space to vocabulary logit space. (2) The logit is adjusted to demote the target token rank to a desired lower rank R. (3) The adjusted logits vector is projected back to hidden space, yielding the updated neuron value.


## Table of Contents
1. [Installation](#installation)
2. [Applying REVS](#applying-revs)
4. [Citation](#citation)


## Installation

To set up your environment for using REVS, we recommend using `conda` to manage Python and CUDA dependencies. We have prepared a script, `setup_conda_env.sh`, which utilizes a YAML file, `revsenv.yml`, to create a new conda environment specifically tailored for REVS. This ensures that all necessary dependencies are correctly installed and configured. Execute the following command to prepare your environment:
```bash
./setup_conda_env.sh
```
This script will create a new conda environment named according to the specifications in `revsenv.yml`. Please ensure that you have Conda installed on your system before running the script.

## Applying REVS
The demo notebook, `notebooks/revs_demo.ipynb`, showcases the unlearning of several organically non-private memorized email addresses through REVS. It evaluates the effectiveness of the unlearning process as well as its robustness against extraction attacks.

Additionally, the code for running the complete suite of experiments, including the baselines of MEMIT and FTL, can be found in the `experiments` directory.

## How to Cite
```bibtex
@article{tomer2024revs,
  title={REVS: Rank Editing in the Vocabulary Space for Unlearning Sensitive Information in Large Language Models},
  author={Ashuach, Tomer and Tutek, Martin and Belinkov, Yonatan},
  journal={},
  volume={},
  year={2024}
}
```