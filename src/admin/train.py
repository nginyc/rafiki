def compute_train_worker_replicas_for_models(models):
    # TODO: Improve provisioning algorithm
    return {
        model : 2 # 2 replicas per model
        for model in models
    }