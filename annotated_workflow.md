preprocess_locomo.py: base namespace in each tool

validation (_validate in ray_trainer.py)
    prepare batch data
    generate_sequences (agent_loop.py) takes in batch data
        every trial:
            _run_agent_loop (agent_loop.py)
                run(tool_agent_loop.py): 1. transforms namespace in each tool from base to trial-level; 2. generate output; 3. extract tool call (max_parallel_calls=1); 4. create tool and execute tool; 5. append content from tool_responses
                compute_score (memory_reward.py)
                pad prompt
                pad response
            _postprocess (agent_loop.py): Process the padded outputs from _run_agent_loop and combine them into a batch.
    test_output_gen_batch_padded -> report metrics

training (start from step 1)
    for epoch in range(self.config.trainer.total_epochs):
        generate_sequences (agent_loop.py) like above, including reward
        compute log probs
        compute values
        compute advantages

        every now and then do validation like above

        update metrics and logging
