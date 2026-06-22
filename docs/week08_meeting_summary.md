# Week 8 Meeting Summary

## Short Update To Say
This week I finalized the Paper 2 analysis and ablation package. I consolidated the Week 6 clean-condition evaluation and the Week 7 robustness results, then added one missing direct ablation for hazard density. The PPO model was kept frozen, the input shape stayed at `50 x 50 x 4`, and there were no training updates.

The main result is that raw PPO is not reliable enough by itself, so it should be treated as an ablation baseline. Raw PPO had 0% clean-condition success and 60% collision rate in both sensing modes. With the safety shield, collision rate dropped to 0% for both monocular and stereo. Monocular PPO+shield had 95.6% clean success, while stereo PPO+shield had 80.8% clean success. Stereo was faster, about 12.26 ms total inference latency, while monocular was slower at about 40.84 ms because MiDaS Small is heavier than StereoSGBM on my machine.

For Week 8, I also ran a 2,000-episode hazard-density ablation using four hazard levels: none, low, moderate, and high. Each mode and hazard level had 250 episodes. Monocular stayed stronger as hazard density increased: it went from 100% success with no/low hazard to 95.2% at moderate and 77.2% at high. Stereo went from 92.0% with no hazard to 90.8% low, 74.0% moderate, and 43.6% high. Collision stayed at 0% because the shield continued to avoid structural obstacles, but hazard exposure and shield interventions increased as hazard density increased.

Week 7 already covered the perception degradation ablations: depth noise, image blur, reduced resolution, and partial occlusion. The strongest failure mode was partial occlusion, especially for stereo. So the final paper narrative is that monocular sensing is slower but more robust in several degraded conditions, stereo is faster but more sensitive to missing or degraded perception, and PPO+shield gives the safety layer needed for reliable collision avoidance.

## Key Numbers

| Result | Monocular | Stereo |
|---|---:|---:|
| Clean PPO+shield success | 95.6% | 80.8% |
| Clean PPO+shield collision | 0.0% | 0.0% |
| Clean total inference latency | 40.84 ms | 12.26 ms |
| High hazard-density success | 77.2% | 43.6% |
| Raw PPO clean success | 0.0% | 0.0% |
| Raw PPO clean collision | 60.0% | 60.0% |

## What I Can Show
- `navigation/results/week08_analysis_ablation/final/visuals/week08_hazard_density_success_rate.png`
- `navigation/results/week08_analysis_ablation/final/visuals/week08_perception_degradation_success.png`
- `navigation/results/week08_analysis_ablation/final/visuals/week08_latency_success_tradeoff.png`
- `navigation/results/week08_analysis_ablation/final/metrics/week08_final_contribution_summary.csv`

## Possible Questions And Answers

**Why is raw PPO not the final controller?**  
Raw PPO failed the clean-condition evaluation: 0% success and 60% collision rate. The paper should use raw PPO only as an ablation showing why the safety shield is needed.

**Did the safety shield change the PPO model?**  
No. The PPO checkpoint and architecture were unchanged. The shield is an evaluation-time safety layer that filters unsafe proposed actions.

**Why is monocular slower than stereo?**  
The monocular baseline uses MiDaS Small, which is a neural monocular depth model. Stereo uses OpenCV StereoSGBM, which ran faster on the local machine.

**Why does stereo degrade more under occlusion and high hazard density?**  
In this controlled setup, stereo-derived risk maps are more sensitive when important structural regions are hidden or degraded. The shield can prevent collisions, but it requires more interventions and has lower goal completion.

**What is the final contribution statement?**  
The paper evaluates the trade-off between stereo and monocular vision for UAV collision avoidance. The results show that a shared PPO policy with a safety shield can use both sensing modes, while monocular offers stronger robustness in these tests at higher computational latency and stereo offers lower latency with higher sensitivity to perception degradation.
