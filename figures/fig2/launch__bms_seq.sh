# logging to file
exec 1> >(tee "results/sbibm_seq/baselines.log") 2> >(tee "results/sbibm_seq/baselines.err" >&2)

method=snpe
num_sims=10_000
seed=4
tabpfnbm +experiment=sbibm_seq_julia method=$method seed=$seed task.num_simulations=$num_sims || true

num_sims=100_000
for seed in {0..4}; do # {0..4}
    echo "--------------------------------"
    echo "Running method=$method num_sims=$num_sims seed=$seed..."
    echo "--------------------------------"
    # each command starts
    # tabpfnbm +experiment=sbibm_seq_core method=$method seed=$seed task.num_simulations=$num_sims || true
    tabpfnbm +experiment=sbibm_seq_julia method=$method seed=$seed task.num_simulations=$num_sims || true
done

for method in snle; do # snpe snle snre
    for num_sims in 100 1_000 10_000 100_000; do #
        for seed in {0..4}; do # {0..4}
            echo "--------------------------------"
            echo "Running method=$method num_sims=$num_sims seed=$seed..."
            echo "--------------------------------"
            # each command starts
            # tabpfnbm +experiment=sbibm_seq_core method=$method seed=$seed task.num_simulations=$num_sims || true
            tabpfnbm +experiment=sbibm_seq_julia method=$method seed=$seed task.num_simulations=$num_sims || true
        done
    done
done

