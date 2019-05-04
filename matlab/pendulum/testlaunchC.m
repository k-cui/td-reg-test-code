parfor trial_idx=1:10
    if trial_idx > 5
        run_single(trial_idx-5, 1, 5)
        runC_single(trial_idx-5, 1, 5)
    else
        run_single(trial_idx, 1, -1)
        runC_single(trial_idx, 1, -1)
    end
end