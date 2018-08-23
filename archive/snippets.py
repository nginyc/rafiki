
    (tuners) = self._get_train_state(train_job)
    tuner = tuners.get(model.name, None)
    
    def _update_train_state(self, train_job, tuners):
        state = self._serialize_train_state({})
        self._db.update_train_job_state(
            train_job,
            state
        )

    def _get_train_state(self, train_job):
        if train_job.state_serialized is None:
            state = self._serialize_train_state({})
            train_job = self._db.update_train_job_state(
                train_job,
                state
            )

        state = self._unserialize_train_state(train_job.state_serialized)
        return ()

    def _serialize_train_state(self, state):
        return dill.dumps(state)

    def _unserialize_train_state(self, state_serialized):
        return dill.loads(state_serialized)
    