# Paper 1 Meeting Script

Good morning sir. For Paper 1, I focused on the specific result organization you asked for: separating the evaluation into three representative synthetic fire environments instead of showing one general comparison folder.

I created three controlled fire-density settings: light/sparse fire, moderate fire, and dense/heavy fire. Then I reran A* and PPO on the same seeded layouts for each environment using the real CNN classifier, the same 50 by 50 grid, the same start and goal setup, and the saved PPO checkpoint.

The new result table is now separated by environment. A* is stronger in the current evaluation: it reaches 100 percent success in light and moderate fire, and 76 percent in dense fire. PPO reaches 64 percent in light fire, 38 percent in moderate fire, and 8 percent in dense fire.

So the main finding is that the classical hazard-aware planner is more reliable in the current three-environment test, while PPO degrades as fire density increases. I think this is still useful for the paper because it gives us a clearer comparison and motivates either a hybrid PPO plus A* strategy or additional PPO training with better dense-fire coverage.

I also updated the episode plots using the style you suggested. The raw episode behavior is still shown in the background, and I added smoothed trend lines with markers so the training trend is easier to read.

One important thing I found while organizing the results is that the old paper text and the saved 3,000-episode logs are not fully aligned. The current saved 3,000-episode PPO record is weaker than the old claims in the paper draft. So I think the final paper should either report these updated numbers honestly, or we should rerun/continue the PPO training before using stronger claims.

The package I prepared includes the three-environment table, the 3,000-episode training summary, the experiment-parameters table, and the updated figures.

## Short Version

I reorganized Paper 1 around light, moderate, and dense fire environments. I reran A* and PPO with the real CNN classifier and the same fixed settings. A* is currently stronger across all three environments, while PPO drops as fire density increases. I also updated the episode plots with raw background curves, smoothed trend lines, and markers. The main issue is that the saved 3,000-episode PPO evidence is weaker than the old paper claims, so we need to decide whether to report these results honestly or rerun PPO training before final submission.

## Likely Questions

### Are these from one map or three environments?

These are from three explicitly controlled fire-density environments: light/sparse, moderate, and dense/heavy fire.

### Did you use the real classifier?

Yes. The run used the actual CNN classifier checkpoint, not synthetic CSV data or a fake hazard map.

### Why did A* beat PPO?

A* has direct hazard-cost planning over the current map, so it is very strong when the classifier-generated hazard map is usable. PPO is more sensitive to the training distribution and did not generalize as well to dense fire in the current checkpoint.

### Does this invalidate PPO?

No. It shows the current PPO checkpoint needs stronger dense-fire training or hybridization. It also gives a useful paper discussion: PPO is adaptive, but A* remains a strong classical baseline.

### What should be changed in the paper?

The old 300-episode table should be replaced with the new 3,000-episode training evidence and the three-environment comparison table. The high success-rate claims should be checked against the saved logs before final submission.

### What is next?

Either rerun PPO training with balanced light/moderate/dense fire scenarios, or frame the final contribution as a comparison showing that A* is currently stronger and a hybrid PPO + A* planner is the next research direction.
