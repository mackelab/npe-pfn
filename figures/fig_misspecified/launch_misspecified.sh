# logging to file
exec 1> >(tee "results/benchmark/misspecified.log") 2> >(tee "results/benchmark/misspecified.err" >&2)

task=misspecified_prior
for seed in 0; do # {0..2}
    for method in filtered_tabpfn; do # npe
        for mu_m in 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
            for tau_m in 0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
                echo "--------------------------------"
                echo "Running task=$task method=$method mu_m=$mu_m tau_m=$tau_m seed=$seed..."
                echo "--------------------------------"
                tabpfnbm method=$method seed=$seed task=$task task.params.mu_m=$mu_m task.params.tau_m=$tau_m || true
            done
        done
    done
done

task=misspecified_likelihood
for seed in 0; do # {0..2}
    for method in npe filtered_tabpfn; do
        for lambda_val in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do # 0 to 1 in steps of 0.1
            for tau_m in 0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do # 0 to 20 in steps of 1
                echo "--------------------------------"
                echo "Running task=$task method=$method lambda_val=$lambda_val tau_m=$tau_m seed=$seed..."
                echo "--------------------------------"
                tabpfnbm method=$method seed=$seed task=$task task.params.lambda_val=$lambda_val task.params.tau_m=$tau_m || true
            done
        done
    done
done
