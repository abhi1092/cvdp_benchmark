# ./run_samples.py -f datasets/cvdp_nonagentic_code_generation_no_commercial.jsonl -l -m anthropic/claude-sonnet-4.5 -n 5 -k 1 -p claude_sonet_4_5

./run_samples.py \
  -f datasets/cvdp_nonagentic_code_generation_no_commercial_test_set.jsonl \
  -l -m vllm/glm-4.7-fp8 \
  -i cvdp_copilot_16qam_mapper_0001 \
  -n 1 -k 1 -p glm_4_7_fp8_test_set