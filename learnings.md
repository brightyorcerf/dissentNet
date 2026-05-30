## Report breakdown

### Section 1: Summary (The "elevator pitch")

What was built: A ResNet-18 that predicts how much humans will disagree on a CIFAR-10 image, rather than just predicting the class label. The output is a 10-way soft distribution over classes, not a single label.

Why it matters: Standard classifiers assume one correct answer. Real annotators disagree, specially on ambiguous images. Modelling that disagreement is useful for active learning, dataset quality analysis, and uncertainty estimation.

What was compared: Three loss functions on identical architecture and training:

- KL divergence
- Jensen-Shannon divergence (JSD)
- Composite loss (KL + entropy mismatch + focal weighting toward hard images)

Headline results to memorise:

- KL model: best overall — test KL of 0.267, Spearman 0.434
- Composite: best at finding the most ambiguous images — P@100 of 0.18
- JSD: worst across everything due to training collapse


### Section 2: Dataset and Sanity Checks

CIFAR-10 test set has 10,000 images. CIFAR-10H collected ~50 human annotations per image, giving you a soft probability distribution over 10 classes for each image instead of one hard label. So instead of "this is a cat," you get "92% of annotators said cat, 4% said dog, etc."

Sanity checks: 
All rows sum to 1.0 within 1e-4 tolerance, the file is well-formed
Majority vote agrees with original CIFAR-10 hard label for 99.21% of images, humans and the dataset are consistent

Split:
From 10,000 images with a fixed random seed:

- 6,000 training
- 2,000 validation
- 2,000 test

Why CIFAR-10H test images for training soft-label model? 
It is training on what was originally a test set, but that's fine because it is solving a different task (distribution prediction vs. classification). The original train/test split was for classification, not disagreement modelling.

The entropy histogram:
Most images have entropy near zero, humans almost unanimously agreed on them. There's a long right tail of genuinely ambiguous images. This class imbalance is exactly why the composite loss's focal weighting was introduced, the model needs to pay extra attention to that rare tail.

The confusion heatmap:
The off-diagonal entries are the interesting ones.

Cat → Dog: ~4% of the time (and dog → cat roughly the same)
Deer → Horse: ~4% roughly symmetric
Automobile → Truck: ~2%

These aren't random errors, they're semantically meaningful confusions between visually similar classes. This validates that the disagreement in CIFAR-10H is real perceptual ambiguity, not noise.

### Section 3: Model Architecture

The core design is CIFAR-style ResNet-18 with:

- 3×3 stem, no max-pool — this is critical
- ResNet stages (layer1 through layer4)
- Global average pooling → 512-d vector
- Linear head → softmax over 10 classes
- Total parameters: 11.17M

Why the CIFAR stem matters:
Standard ResNet-18 (designed for ImageNet) uses a 7×7 stride-2 conv followed by max-pooling. On 224×224 ImageNet images that's fine. On 32×32 CIFAR images it would destroy most of the spatial information before the network even starts learning. The 3×3 stem preserves that spatial resolution.

Same architecture, different losses:
All three models — KL, JSD, composite — are identical in architecture. The only thing that differs is the training loss. This is good experimental design: any performance difference is attributable to the loss function, not the model.

Pretraining:
The backbone was first pretrained on all 50,000 CIFAR-10 hard-label images for 30 epochs, reaching 99.4% classification accuracy. Then the head was replaced and fine-tuned on soft labels. This is transfer learning within the same domain, the backbone already knows what cats and trucks look like; fine-tuning teaches it to model disagreement.

### Section 4: Training Behaviour

All three models used the same early stopping rule: stop if validation KL doesn't improve for 8 epochs, hard cap at 50. All three stopped well before 50, which itself tells something: the task is hard to learn from 6,000 soft-label images.

#### KL Model - Classic Overfitting

Training loss drops nicely: 0.36 → ~0.10 over the epochs
But validation KL hits its best at epoch 2 (0.230), then drifts back up to ~0.27.

This is textbook early overfitting, the model memorises training distributions instead of generalising.

The "best checkpoint" is essentially the very first one, which is a red flag about training set size (addressed in section 8)

Why it overfits so fast:
- The soft-label training split is only 6,000 images 
- Not enough signal to support many epochs of distribution learning
- Most images have near-zero entropy
- The model has very few genuinely ambiguous examples to learn from

#### JSD Model - Degenerate Collapse

Training loss goes flat near zero from epoch 1
Validation KL worsens over training: 0.417 → 0.531
The model found a shortcut, it satisfies the JSD objective mathematically without actually learning to predict disagreement

This is a known failure mode with symmetric divergences on imbalanced distributions

Why JSD collapses? 
- JSD is bounded between 0 and 1 and symmetric
- the model can find a near-zero-loss solution by outputting near-uniform or near-peaked predictions that still look "close" under JSD

#### Composite Model - Noisy but Salvageable

Training is noisier than KL,loss jumps around throughout
Validation KL hits 0.275 at epoch 3, then oscillates between 0.29–0.39

Early stopping caught a good checkpoint despite instability
The noise is likely because the focal weighting term is too aggressive, it keeps pushing the model hard toward ambiguous examples

"Early stopping did its job here. The instability is a known tradeoff of focal losses; you're amplifying a sparse signal (ambiguous images), which introduces gradient noise."

### Section 5: Test Set Performance

Performance table for DissentNet. This benchmark highlights how different loss functions perform across distributional metrics (KL, JSD, Cosine Similarity, Spearman Rank Correlation, and Precision at $K$).

| Loss Function | KL ↓ | JSD ↓ | Cosine ↑ | Spearman ↑ | P@100 ↑ | P@500 ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KL (Our Model)** | **0.267** | **0.053** | **0.934** | 0.434 | 0.160 | **0.516** |
| **JSD** | 0.459 | 0.056 | 0.927 | 0.365 | 0.130 | 0.464 |
| **Composite** | 0.320 | 0.060 | 0.922 | **0.423** | **0.180** | 0.496 |

- KL_mean — How far, on average, is your predicted distribution from the human distribution? Lower = better. KL wins this clearly.
- JSD_mean — Same idea but symmetric and bounded. Interestingly JSD the model doesn't win this despite being trained on JSD loss — further evidence of its collapse.
- Cosine similarity — Are the shapes of predicted and true distributions pointing in the same direction? All three are decent (0.92+), meaning all models get the direction roughly right.
- Spearman correlation — If you rank all 2,000 test images by predicted uncertainty and by true uncertainty, how well do those rankings agree? KL's 0.434 means it's best at ordering images by how much humans disagreed.
- P@100 / P@500 — "Of the top-K images the model flagged as most ambiguous, what fraction were genuinely ambiguous to humans?" P@100 is the harder, stricter version. Composite wins P@100 (0.18) despite losing on bulk metrics.

##### 5.1 KL — Best Overall
Wins 5 out of 6 metrics. Best for general distribution matching. If your use case is "approximate the full shape of human disagreement across all images," use this.

##### 5.2 Composite — Best for Finding the Hard Cases
Loses to KL by 16–20% on bulk metrics, but beats KL on P@100 (0.18 vs 0.16). The focal weighting is working — it sacrifices average performance to get better at the rare, genuinely ambiguous images. This is exactly what you'd want for active learning pipelines, where you need to find the images worth getting more human labels for.

#### 5.3 JSD — Worst Across the Board
Worst on every single metric. The training collapse from 4b shows up directly in test results.

#### The scatter plot (predicted vs. true entropy):
The cloud of points sits above the diagonal, meaning the model systematically over-predicts uncertainty. Easy images get assigned more uncertainty than they deserve. This is the model's main failure mode and worth mentioning proactively.

### Section 6: Robustness Analysis

#### 6.1 Annotator Subsampling — "The Illusion of Certainty"

Simulated having fewer annotators by subsampling the label distributions, using n ∈ {5, 10, 20, 30, 40, 50} annotators per image, then re-evaluated the composite model.

As annotator count drops, label entropy falls, the distribution looks more confident even though the image hasn't changed. Rare dissenting votes disappear, making the crowd look more certain than it actually is.

The model's KL rises (gets worse) not because the model changed, but because the target got noisier/less faithful (comparing against a degraded ground truth).

The practical lesson: Always benchmark soft-label models against the largest annotator pool you have. Small crowds give you falsely confident-looking ground truth, which makes your model look worse than it is.

This is an insight about soft-label benchmarking in general, not just this project.

#### 6.2 OOD Corruptions

Three corruption families at 5 severity levels each:
- Gaussian noise
- Gaussian blur
- Contrast reduction

What happened:
Blur and contrast behave as expected — as severity increases, predicted entropy increases. The model correctly becomes more uncertain as the image degrades. This is the right behaviour.

Gaussian noise is different and problematic. Uncertainty peaks at severity 2, then falls at higher severities. At maximum noise, the model becomes confidently wrong — it picks a class with high confidence even though the image is essentially unrecognisable.

Why does this happen? High-frequency Gaussian noise at severe levels may accidentally resemble texture patterns the backbone learned to associate with specific classes. The model latches onto that noise as a confident signal.

Blur and contrast degrade the image in ways that preserve semantic ambiguity, so the model responds correctly. Gaussian noise at high severity creates artificial texture patterns that the model misinterprets as confident evidence, this is the next failure mode to address.

### Section 7: Qualitative Analysis

- Grad-CAM overlays: Grad-CAM visualises where in the image the model is paying attention when making predictions.
- Easy images (low entropy): Attention focuses tightly on the main object — the cat, the truck, the frog. Clean, localised activation.
- Ambiguous images (high entropy): Attention spreads across multiple regions of the image. The model isn't sure what to focus on because the image genuinely contains ambiguous visual information.

This is the qualitative signature that the model has actually learned something meaningful about disagreement, it's not just outputting uniform distributions, it's responding to genuine visual ambiguity in a spatially meaningful way.

Failure cases plot:
The worst failures are shown with blue bars (human distribution) vs. orange bars (model prediction).
The pattern: humans found the image easy (sharp, confident blue bar), but the model hedged (spread-out orange bars).
This is the opposite of a standard classifier's failure mode. Normal classifiers are overconfident. DissentNet is over-uncertain, it spreads probability mass on images that are actually unambiguous to humans. This connects directly to the scatter plot from section 5 where the cloud sits above the diagonal (over-predicting entropy).

### Limitations

KL hit its best score at epoch 2 and got worse from there, 
the 6,000-image training set is just too small to support more than a couple of useful epochs, so stronger regularisation or aggressive LR decay are needed to push past that. JSD's training loss flatlining near zero from epoch 1 is almost certainly a degenerate optimum; a log-space rewrite would probably help. And the composite
loss is jumpy enough (Figure 4c) that we suspect the focal weight is too strong, easing the entropy term in with a warmup, or just lowering gamma, would likely smooth things out.

### Conclusion

On CIFAR-10H, KL-trained ResNet-18 is our best overall model, it's test KL is 0.267 and Spearman of 0.434.

If it's about catching the most ambiguous images for active learning, use the composite loss. It wins P@100 with 0.18. 

The subsampling experiment shows that small annotator pools give you a falsely confident-looking ground truth, so always benchmark against the largest crowd you have. The corruption analysis was a mixed bag, the model handles blur and contrast well, but on high-noise images it gets confidently wrong instead of uncertain.

## Defending the project

What's the biggest limitation of this whole project?
A: Training set size. 6,000 images with a heavy class imbalance (most near-zero entropy) means the model can't learn the ambiguous tail well. The KL model's best checkpoint being epoch 2 is a direct symptom of this.

If you had to pick one model to deploy, which would it be and why?
A: Depends on use case. For general distribution matching, KL model. For active learning (finding images worth relabelling), composite model because of P@100.

Why does the model over-predict uncertainty systematically?
A: The training distribution is heavily skewed toward low-entropy images. The model sees relatively few genuinely ambiguous examples, so when it encounters something unfamiliar it defaults to spreading probability rather than committing: a conservative bias baked in by the data imbalance.

What would a natural next step be?
A: Larger annotator pools for ground truth, stronger regularisation for KL fine-tuning, fixing JSD's degenerate optimum with a log-space reformulation, and addressing the Gaussian noise failure mode with adversarial or noise-specific training.

### On architectural choices

Why ResNet-18 and not a larger model like ResNet-50 or ViT?
A: CIFAR-10 images are 32×32, a ViT needs much larger inputs to work well, and ResNet-50 would overfit even faster on 6,000 images. ResNet-18 at 11.17M parameters is already borderline large for this dataset size.

Why the CIFAR stem (3×3, no max-pool) and not the standard ImageNet stem?
A: Standard stem uses 7×7 stride-2 conv + max-pool, which downsamples aggressively. On 32×32 images you'd lose most spatial information before the first residual block even runs.

Why a linear head and not an MLP head?
A: The paper mentions "Linear / MLP / Temp" as head options. A linear head is simplest and least likely to overfit given the small training set. An MLP head adds capacity but also adds overfitting risk, not worth it when your best checkpoint is already epoch 2.

Why softmax output and not sigmoid per class?
A: Need a valid probability distribution that sums to 1, matching the human annotation distribution. Sigmoid outputs are independent and don't sum to 1, which is the wrong inductive bias for this task.

Why global average pooling before the head?
A: Standard for classification ResNets. Reduces spatial feature maps to a single vector without adding parameters. Alternatives like flattening would massively increase parameter count and overfit.

Why KL divergence and not cross-entropy?
A: Standard cross-entropy assumes a one-hot hard label. KL divergence generalises it to soft distributions, it measures how much information is lost using your predicted distribution instead of the human one. For soft labels, KL is the natural choice.

Why JSD as a candidate at all?
A: JSD is symmetric (KL is not) and bounded between 0 and 1, making it more numerically stable in theory. It was a reasonable hypothesis that symmetry would help. It didn't, the model found a degenerate optimum.

What other losses could you have tried?

- Wasserstein / Earth Mover's Distance: respects the geometry between classes (cat and dog are "closer" than cat and airplane). Could be meaningful here since the confusions are semantically structured.
- MSE on the distribution: simple but treats all class errors equally, no probabilistic grounding.
- Label smoothing variants: blending hard labels with soft ones, but you already have soft labels so this is redundant.
- Focal loss on entropy: you effectively did this with the composite, but a pure focal formulation without the KL component could be interesting.

Why did JSD collapse specifically?
A: JSD is bounded and symmetric, so gradients vanish when predictions are close to uniform. The model found a near-uniform output that minimised JSD on training data without actually learning class-specific disagreement patterns.

What would fix JSD?
A: Log-space reformulation to avoid numerical issues, or adding a KL regulariser to prevent the degenerate solution.

Why pretrain on hard labels first?
A: The soft-label training set is only 6,000 images. Training from scratch on that would never converge well. Pretraining on 50,000 hard-label CIFAR-10 images gives the backbone solid visual representations, it already knows what a cat looks like. Fine-tuning then only needs to teach disagreement modelling, which is a much easier task.

Why early stopping on validation KL with patience 8?
A: KL is the primary evaluation metric, so it's the right signal to monitor. Patience 8 is generous enough to survive oscillations (relevant for composite) without wasting compute.

Why a 6k/2k/2k split and not something like 8k/1k/1k?
A: Fixed seed ensures reproducibility. The 2k test set gives you enough images for statistically meaningful Spearman and Precision@K evaluation. Larger training set would have helped but you'd lose evaluation reliability.

Why not use the full 50,000 CIFAR-10 training images for soft-label training?
A: Those images don't have soft labels.
CIFAR-10H only covers the original 10,000 test images. You can't manufacture soft labels from hard ones without additional annotation.

Why the same architecture for all three loss variants?
A: Clean ablation, isolates the effect of the loss function. If architectures differed, you couldn't attribute performance differences to the loss.

Why use Spearman correlation and not Pearson?
A: Spearman is rank-based and doesn't assume linearity. Given the heavily skewed entropy distribution (most images near zero, long tail), Pearson would be distorted by outliers. Spearman just asks: do you rank images in the same uncertainty order as humans?

Why Precision@K and not just mean KL?
A: Mean KL averages over all images, including the easy ones which dominate. P@K specifically evaluates performance on the high-disagreement tail — the practically important cases for active learning. A model could have great mean KL while being terrible at finding ambiguous images.

Why P@100 and P@500 specifically?
A: P@100 is strict — top 5% of the test set. P@500 is the top 25%. Together they show whether the model's uncertainty rankings hold up at different recall levels. Composite wins P@100 but loses P@500, suggesting its advantage is concentrated at the very top of the uncertainty ranking.

Why cosine similarity as a metric?
A: Complements KL, it measures directional alignment of distributions regardless of magnitude. A model could have poor KL but still get the shape right, or vice versa.

The KL model overfits at epoch 2, doesn't that mean it barely learned anything?
A: The backbone learned everything during pretraining. Fine-tuning just adapts the head to soft labels, and that happens quickly. Two epochs of meaningful gradient updates on 6,000 images is actually reasonable.

Composite loses on mean KL but wins P@100, how do you reconcile that?
A: Different objectives optimise for different things. Focal weighting deliberately sacrifices average performance to concentrate capacity on the ambiguous tail. It's a precision-recall tradeoff, you're trading bulk accuracy for better retrieval of the hardest cases.

JSD_mean is similar across all three models (0.053, 0.056, 0.060) even though KL_mean varies a lot, why?
A: JSD is bounded and symmetric, so it compresses differences. KL is unbounded and asymmetric, so it's more sensitive to cases where the predicted distribution differs significantly from the human one. KL_mean is a more discriminating metric here.

What's the practical application of a model like this?
A: Active learning — find images worth getting more human labels for. Dataset quality auditing — identify where your labels are unreliable. Uncertainty-aware inference — don't report a confident prediction on images humans themselves disagree about.

How is this different from standard uncertainty estimation (dropout, ensembles)?
A: Standard uncertainty methods estimate model uncertainty, how confident is the network given its training. DissentNet estimates human uncertainty — how much do annotators disagree. These are different things. A model can be very confident about an image that humans find ambiguous, and vice versa.

Could this approach generalise beyond CIFAR-10?
A: Yes, anywhere you have multi-annotator soft labels: medical imaging, content moderation, sentiment analysis. The architecture is dataset-agnostic; the key requirement is a soft label dataset with enough annotators per item.

Is a Spearman of 0.434 actually good?
A: Honestly it's moderate. It's better than random (0) and shows meaningful signal, but far from perfect (1.0). The task is genuinely hard — predicting exactly how 50 humans will split on a 32×32 image from visual features alone is an extremely noisy signal to regress on.

## Codebase

requirements.txt — Python dependencies. PyTorch, torchvision, scipy (for Spearman), matplotlib, numpy etc.
README.md — Project overview and how to run it.
learnings.md — Sounds like your team documented insights/observations as you went. Good thing to revisit before viva.
.gitignore — Tells git to ignore checkpoints, large files etc.

configs/default.yaml
Central config file — stores all hyperparameters in one place: learning rate, batch size, patience, epochs, seed, split sizes, focal gamma value for composite loss. If asked in viva "what were your hyperparameters," this is where they all live.

src/ — The core library
These are the reusable modules that scripts call into.
data.py — Loads CIFAR-10H, handles the 6k/2k/2k split with fixed seed, returns dataloaders. The sanity checks (rows sum to 1, majority vote agreement) probably live here too.
models.py — Defines the CIFAR-style ResNet-18 — the 3×3 stem, no max-pool, linear head, softmax output. The 11.17M parameter architecture from section 3.
losses.py — The three loss functions: KL divergence, JSD, and the composite (KL + entropy mismatch + focal weighting). Directly section 4 relevant — the only thing that differs between your three training runs.
evaluate.py — Computes the full metric suite: KL_mean, JSD_mean, cosine similarity, Spearman correlation, P@100, P@500. Directly section 5 relevant — this is what produced your results table.
train.py — Training loop: forward pass, loss computation, validation KL monitoring, early stopping logic (patience=8). Core section 4 file — the training curves you see in 4a/4b/4c come from this.
explain.py — Grad-CAM implementation. Generates the attention overlays from section 7.
robustness.py — Applies corruptions (Gaussian noise, blur, contrast) at severity levels 1–5 and re-evaluates. Section 6.2.
viz.py — All plotting code. Every figure in the report was generated from here.
utils.py — Shared utility functions — probably seed setting, checkpoint saving/loading, directory management.
__init__.py — Makes src a Python package so scripts can do from src.models import ...

scripts/ — The pipeline, run in order
01_prepare_data.py — Downloads/loads CIFAR-10H, runs sanity checks, creates the splits, saves splits.npz. Section 2.
02_train_all_losses.py — Trains all three models (KL, JSD, composite) by calling src/train.py with different loss configs. Saves the three checkpoints. Section 4's entry point.
03_run_ablations.py — Likely the annotator subsampling experiment from section 6.1.
04_evaluate_all.py — Loads all three best checkpoints, runs src/evaluate.py on the 2k test set, saves loss_comparison.csv. Section 5's entry point.
05_robustness.py — Runs the OOD corruption analysis, saves robustness_B_corruptions_compo...csv. Section 6.2.
06_explain.py — Runs Grad-CAM on selected images, saves gradcam_low_high_composite.png. Section 7.

outputs/ — Everything generated
checkpoints/

pretrain_cifar10.pt — Backbone after 30 epochs on 50k hard labels, 99.4% accuracy
best_kl.pt — Best checkpoint for KL model (epoch 2 per section 4a)
best_jsd.pt — Best checkpoint for JSD model
best_composite.pt — Best checkpoint for composite model

tables/ — Section 4 and 5 critical files

loss_comparison.csv — This is your section 5 results table. KL_mean, JSD_mean, cosine, Spearman, P@100, P@500 for all three models.
data_sanity_checks.json — Output of the section 2 sanity checks (row sums, majority vote agreement 99.21%)
data_alignment_check.json — Verifies the split indices are consistent
failure_cases_composite.csv — The worst predictions from section 7, ranked by |H(p) - H(q)|
robustness_A_subsampling_compo...csv — Section 6.1 annotator subsampling results table
robustness_B_corruptions_compo...csv — Section 6.2 corruption severity results
manual_disagreement_inspection...csv — Probably a hand-curated list of interesting ambiguous cases
per_class_entropy.npy — Per-class entropy values, used for the confusion heatmap analysis
soft_confusion.npy — The full 10×10 soft confusion matrix from section 2
splits.npz — The saved train/val/test index splits with fixed seed
true_entropy.npy — Ground truth entropy values for all test images, used for Spearman computation
param_counts.csv — Parameter count breakdown (confirms 11.17M)

logs/

training_histories.json — Section 4's data source. Train loss and validation KL per epoch for all three runs. The training curve plots (4a, 4b, 4c) were generated directly from this file.

figures/ — Section 4 and 5 relevant ones

training_curves_kl.png — Figure 4a
training_curves_jsd.png — Figure 4b (with the flat training loss)
training_curves_composite.png — Figure 4c
loss_comparison_KL.png — The bar chart comparing KL_mean across three models (section 5)
loss_comparison_spearman.png — The Spearman bar chart (section 5)
entropy_scatter_kl.png — Predicted vs. true entropy scatter plot for KL model (section 5)
entropy_scatter_jsd.png — Same for JSD
entropy_scatter_composite.png — Same for composite
gradcam_low_high_composite.png — Section 7 Grad-CAM overlays
failure_cases_composite.png — Section 7 worst case plots
data_entropy_histogram.png — Section 2 entropy histogram
data_soft_confusion.png — Section 2 confusion heatmap
architecture.png — Section 3 architecture diagram
robustness_corruption_response_...png — Section 6.2 corruption plot
