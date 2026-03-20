for num_simulations in 100 1000 2000; do # 500
    for noise_scale in 1.0 2.0; do
        for feature_scale in 1.0 2.0; do
            # for dim in 5 10; do # 20
            echo "--------------------------------"
            echo "Running noise_scale=$noise_scale feature_scale=$feature_scale dim=$dim num_simulations=$num_simulations..."
            echo "--------------------------------"
            tabpfnbm +experiment=flexible_linear task.params.noise_scale=$noise_scale task.params.feature_scale=$feature_scale task.num_simulations=$num_simulations || true # task.dim=$dim
            tabpfnbm +experiment=flexible_linear_tabpfn task.params.noise_scale=$noise_scale task.params.feature_scale=$feature_scale task.num_simulations=$num_simulations || true # task.dim=$dim
            # done
        done
    done
done
